"""
V-JEPA 2 wrapper for Track B (open world).

Loads a V-JEPA 2 SSv2-finetuned checkpoint (default
``facebook/vjepa2-vitl-fpc16-256-ssv2``), swaps the 174-class head to the
challenge's ``num_classes`` (33), and exposes the standard
``(B, T, C, H, W) -> (B, num_classes)`` interface used everywhere in this repo.

Also exposes ``get_param_groups_llrd`` for layer-wise LR decay: head gets full
LR, each transformer block earlier in the network gets ``base_lr × decay^d``
where ``d`` is the depth-to-head. Standard for ViT fine-tuning (DeiT, BEiT, MAE).

Why V-JEPA 2:
    Meta's V-JEPA 2 (2025) is the current public state-of-the-art on
    Something-Something V2. The pretraining is self-supervised on a very large
    video corpus, and the finetune is on the exact data distribution as the
    challenge. Expect a clear lift vs TimeSformer-base / VideoMAE-base.

Input adaptation:
    The fpc16 / 256 checkpoint expects 16 frames at 256x256. The repo's
    ``processed_data`` provides 4 frames at 224x224. The wrapper handles both:

    - Temporal: replicates each input frame ``native_frames // T`` times along
      the time axis. Order is preserved (no shuffle). With T=4 and native=16,
      ``[f0, f1, f2, f3] -> [f0,f0,f0,f0, f1,f1,f1,f1, f2,f2,f2,f2, f3,f3,f3,f3]``.
      This is the safest "downscale": no need to monkey-patch internal HF
      attributes (which have moved between transformers versions).

    - Spatial: bilinearly upsamples ``(B*T, C, 224, 224) -> (B*T, C, 256, 256)``
      if the input size doesn't match ``image_size``.

Note: ImageNet normalization (``build_transforms(use_imagenet_norm=True)``) is
correct for V-JEPA 2 checkpoints.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default LoRA target modules for V-JEPA 2 attention. Covers both the
# llama-style naming (``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``) and
# the bert-style naming (``query`` / ``key`` / ``value``) so the wrapper
# works regardless of which transformers release implemented the V-JEPA 2
# port. peft matches by suffix and silently skips names that don't exist,
# so listing both forms is safe.
_DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query", "key", "value",
)


class VJEPA2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_id: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        num_frames: int = 4,
        image_size: int = 256,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False,
        # ----- LoRA (parameter-efficient fine-tuning) ---------------------- #
        # When ``use_lora=true``, the backbone is frozen and small low-rank
        # adapter matrices are inserted in the attention projections. The
        # classifier head stays fully trainable (via peft's modules_to_save).
        # Defaults are off so existing checkpoints / commands are unaffected.
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Sequence[str]] = None,
        lora_bias: str = "none",
    ) -> None:
        super().__init__()

        # Local import so Track A users don't need transformers installed.
        try:
            from transformers import AutoModelForVideoClassification
        except ImportError as e:
            raise ImportError(
                "V-JEPA 2 requires a recent `transformers` (>= 4.55). "
                "Run `uv sync` or `uv pip install -U transformers`."
            ) from e

        if not pretrained:
            raise ValueError(
                "V-JEPA 2 from-scratch is not useful here — set "
                "`model.pretrained=true` or use cnn_baseline / cnn_transformer "
                "for from-scratch ablations."
            )

        self.model = AutoModelForVideoClassification.from_pretrained(
            model_id,
            num_labels=int(num_classes),
            ignore_mismatched_sizes=True,
        )

        # The native frame count attribute name has varied across V-JEPA 2
        # transformers releases ("frames_per_clip", "num_frames"). Check both.
        cfg = self.model.config
        native_frames = int(
            getattr(cfg, "frames_per_clip", getattr(cfg, "num_frames", 16))
        )
        self.native_frames = native_frames
        self.target_frames = int(num_frames)
        self.image_size = int(image_size)

        if self.native_frames % self.target_frames != 0:
            raise ValueError(
                f"V-JEPA 2 native num_frames={self.native_frames} is not a "
                f"multiple of dataset num_frames={self.target_frames}. Pick a "
                "num_frames that divides the native count (e.g. 4 or 8 for 16)."
            )
        self._repeat = self.native_frames // self.target_frames

        # The forward kwarg for video has moved from `pixel_values` to
        # `pixel_values_videos` in newer transformers. Detect once at init.
        params = inspect.signature(self.model.forward).parameters
        self._video_kwarg = (
            "pixel_values_videos" if "pixel_values_videos" in params else "pixel_values"
        )

        if use_lora and freeze_backbone:
            # Both flags are about parameter efficiency; LoRA already freezes
            # everything except its adapters + the head, so the freeze_backbone
            # toggle is redundant in that case. Warn rather than silently pick.
            print(
                "WARNING: both `use_lora=true` and `freeze_backbone=true` are set. "
                "LoRA already freezes the backbone — ignoring `freeze_backbone`.",
                flush=True,
            )

        if use_lora:
            self._apply_lora(
                lora_r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(lora_target_modules)
                if lora_target_modules is not None
                else list(_DEFAULT_LORA_TARGET_MODULES),
                bias=str(lora_bias),
            )
        elif freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

        # Gradient checkpointing: trades compute for VRAM by re-running the
        # forward of each transformer block during backward instead of caching
        # activations. Essential for the ViT-g (1B) variant at 64×384² inputs.
        # `use_reentrant=False` is the future-proof path (the True default is
        # deprecated and breaks under torch.compile / DDP find_unused_params).
        if use_gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
            except AttributeError:
                pass

    def _apply_lora(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: List[str],
        bias: str,
    ) -> None:
        """Wrap ``self.model`` with a peft LoraConfig.

        Backbone params are frozen by peft, low-rank A/B adapters are inserted
        on each module whose name suffix matches ``target_modules``, and the
        ``classifier`` is kept fully trainable via ``modules_to_save`` (peft
        also handles its serialization/deserialization).
        """
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError(
                "use_lora=true requires the `peft` package. Run `uv sync` to "
                "install it (it is declared in pyproject.toml)."
            ) from e

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
            # The classifier head must stay fully trainable (not LoRA-wrapped)
            # because we're remapping 174 SSv2 logits -> 33 challenge logits.
            modules_to_save=["classifier"],
        )
        self.model = get_peft_model(self.model, lora_cfg)

        # Sanity print so the user can confirm LoRA actually attached.
        try:
            self.model.print_trainable_parameters()
        except AttributeError:
            pass

    def _count_encoder_layers(self) -> int:
        """Count transformer blocks by inspecting parameter names.

        Robust to peft's ``base_model.model.…`` prefix when LoRA is active.
        """
        layer_nums = set()
        for name, _ in self.model.named_parameters():
            m = re.search(r"\.layer\.(\d+)\.", name) or re.search(
                r"\.layers\.(\d+)\.", name
            )
            if m:
                layer_nums.add(int(m.group(1)))
        return max(layer_nums) + 1 if layer_nums else 24  # ViT-L default

    def get_param_groups_llrd(
        self,
        base_lr: float,
        weight_decay: float,
        decay_rate: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Layer-wise LR decay param groups.

        Head gets ``base_lr``. Encoder block i gets ``base_lr × decay^(N - i)``
        where N is the deepest block. Embeddings get the smallest LR. Biases /
        norm / position-embed params get weight_decay=0 (ViT standard).
        """
        num_layers = self._count_encoder_layers()
        max_depth = num_layers + 1  # head sits one step beyond the last block

        def layer_id(name: str) -> int:
            # Higher id = closer to head = larger LR.
            if "classifier" in name or "pooler" in name:
                return max_depth
            m = re.search(r"\.layer\.(\d+)\.", name) or re.search(
                r"\.layers\.(\d+)\.", name
            )
            if m:
                return int(m.group(1)) + 1
            # Final norm of the encoder, treat as the last block.
            if "encoder.layernorm" in name.lower() or name.endswith(".norm.weight"):
                return num_layers
            # Embeddings, patch_embed, pos_embed → deepest depth from head.
            return 0

        # Group params by (layer_id, decay/nodecay).
        groups: Dict[tuple, List[nn.Parameter]] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            lid = layer_id(name)
            # ViT convention: no weight decay on biases, LayerNorm, pos embeds.
            no_decay = (
                param.ndim <= 1
                or name.endswith(".bias")
                or "layernorm" in name.lower()
                or "pos_embed" in name.lower()
                or "position_embeddings" in name.lower()
                or "cls_token" in name.lower()
            )
            groups.setdefault((lid, no_decay), []).append(param)

        param_groups: List[Dict[str, Any]] = []
        for (lid, no_decay), params in sorted(groups.items()):
            multiplier = decay_rate ** (max_depth - lid)
            param_groups.append(
                {
                    "params": params,
                    "lr": base_lr * multiplier,
                    "weight_decay": 0.0 if no_decay else weight_decay,
                }
            )
        return param_groups

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, C, H, W) -> logits (B, num_classes)."""
        B, T, C, H, W = video.shape

        # Temporal: replicate frames to reach the native count, preserving order.
        if T != self.native_frames:
            video = video.repeat_interleave(self._repeat, dim=1)

        # Spatial: resize to the model's expected input size.
        if H != self.image_size or W != self.image_size:
            _, T2, C2, _, _ = video.shape
            flat = video.reshape(B * T2, C2, H, W)
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            video = flat.reshape(B, T2, C2, self.image_size, self.image_size)

        outputs = self.model(**{self._video_kwarg: video})
        return outputs.logits


class VJEPA2ZeroShotClassifier(nn.Module):
    """V-JEPA 2 used **zero-shot**: the original SSv2 (174-class) head is kept
    intact, and the model's logits are sliced down to the challenge's subset
    via a pre-computed index mapping. No training needed.

    The mapping is provided by the caller — usually built by matching
    challenge class folder names against ``hf_model.config.id2label``.

    Frame-replication and spatial-resize mechanics mirror ``VJEPA2Classifier``
    so the model still consumes the project's standard ``(B, T, C, H, W)``
    layout with ``T == dataset.num_frames``.
    """

    def __init__(
        self,
        class_indices: List[int],
        model_id: str = "facebook/vjepa2-vitg-fpc64-384-ssv2",
        num_frames: int = 4,
        image_size: int = 384,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoModelForVideoClassification
        except ImportError as e:
            raise ImportError(
                "V-JEPA 2 requires a recent `transformers` (>= 4.55). "
                "Run `uv sync` or `uv pip install -U transformers`."
            ) from e

        # IMPORTANT: no `num_labels` / `ignore_mismatched_sizes` override here.
        # We need the original SSv2 head intact so the pretrained logits stay
        # meaningful — that's the whole point of zero-shot.
        self.model = AutoModelForVideoClassification.from_pretrained(model_id)

        self.register_buffer(
            "ssv2_indices", torch.tensor(class_indices, dtype=torch.long)
        )

        cfg = self.model.config
        native_frames = int(
            getattr(cfg, "frames_per_clip", getattr(cfg, "num_frames", 16))
        )
        if native_frames % int(num_frames) != 0:
            raise ValueError(
                f"V-JEPA 2 native num_frames={native_frames} is not a multiple "
                f"of dataset num_frames={num_frames}."
            )
        self.native_frames = native_frames
        self.target_frames = int(num_frames)
        self._repeat = self.native_frames // self.target_frames
        self.image_size = int(image_size)

        params = inspect.signature(self.model.forward).parameters
        self._video_kwarg = (
            "pixel_values_videos" if "pixel_values_videos" in params else "pixel_values"
        )

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, C, H, W) -> logits sliced to the 33-class subset."""
        B, T, C, H, W = video.shape

        if T != self.native_frames:
            video = video.repeat_interleave(self._repeat, dim=1)

        if H != self.image_size or W != self.image_size:
            _, T2, C2, _, _ = video.shape
            flat = video.reshape(B * T2, C2, H, W)
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            video = flat.reshape(B, T2, C2, self.image_size, self.image_size)

        outputs = self.model(**{self._video_kwarg: video})
        # Keep only the challenge's 33 columns from the native 174-class logits.
        return outputs.logits[:, self.ssv2_indices]
