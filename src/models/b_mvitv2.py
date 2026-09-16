"""
MViT / MViTv2 wrapper for Track B (open world).

Multiscale Vision Transformer (Fan et al. CVPR 2021 / ICCV 2021 v2). Unlike
ViT-style models, MViT uses **stage-wise spatial pooling** (like a CNN
pyramid) inside the transformer blocks, producing a hierarchical feature map.
This makes its inductive bias quite different from V-JEPA 2 / VideoMAE /
TimeSformer, which is precisely why it's worth adding to an ensemble — its
errors are decorrelated from pure ViT-based models, even when their individual
top-1 numbers are similar.

Available HuggingFace checkpoints (verify availability before training):

    facebook/mvit-base-finetuned-ssv2     ~ 36M params  (MViT v1, ~67% SSv2)
    facebook/mvit-large-finetuned-ssv2    ~ 213M params (may not be public)
    facebook/mvitv2-base-finetuned-ssv2   (MViT v2, if released)
    facebook/mvitv2-large-finetuned-ssv2  (MViT v2 Large)

The wrapper is **generic** in ``model_id``: pick whichever HuggingFace-hosted
MViT/MViTv2 SSv2 checkpoint suits your VRAM. If a chosen ``model_id`` is not
hosted on HF (e.g. only available via PyTorchVideo or the official repo),
the load will fail and you'll need a different integration path.

Input adaptation: same logic as ``b_vjepa2.py``:
  - Temporal: replicate frames via ``repeat_interleave`` to match the
    model's native ``num_frames`` (typically 16).
  - Spatial: bilinear-resize to the model's native ``image_size``
    (typically 224 for the Base variant).

Normalization: ImageNet stats (``build_transforms(use_imagenet_norm=True)``).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MViTv2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_id: str = "facebook/mvit-base-finetuned-ssv2",
        num_frames: int = 4,
        image_size: int = 224,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # Local import so Track A users don't need transformers installed.
        try:
            from transformers import AutoModelForVideoClassification
        except ImportError as e:
            raise ImportError(
                "MViT requires `transformers`. Run `uv sync` or "
                "`uv pip install -U transformers`."
            ) from e

        if not pretrained:
            raise ValueError(
                "MViT from-scratch is not useful on this dataset — set "
                "`model.pretrained=true`, or use cnn_baseline / cnn_transformer / "
                "cnn_lstm / two_stream_transformer for from-scratch ablations."
            )

        self.model = AutoModelForVideoClassification.from_pretrained(
            model_id,
            num_labels=int(num_classes),
            ignore_mismatched_sizes=True,
        )

        # Native temporal length: HF MViT configs expose ``num_frames`` (older
        # MViT) or ``temporal_size`` (some MViTv2 builds). Default 16.
        cfg = self.model.config
        native_frames = int(
            getattr(cfg, "num_frames", getattr(cfg, "temporal_size", 16))
        )
        # Native spatial size: ``image_size`` on most configs, fallback 224.
        native_image_size = int(getattr(cfg, "image_size", 224))

        self.native_frames = native_frames
        self.target_frames = int(num_frames)

        if self.native_frames % self.target_frames != 0:
            raise ValueError(
                f"MViT native num_frames={self.native_frames} is not a multiple "
                f"of dataset num_frames={self.target_frames}. Pick a num_frames "
                "that divides the native count (e.g. 4 or 8 for 16)."
            )
        self._repeat = self.native_frames // self.target_frames

        # If user passed a different image_size, override the wrapper's resize
        # target; otherwise trust the model's native size.
        self.image_size = int(image_size) if image_size else native_image_size

        # Forward kwarg has migrated across transformers versions:
        # ``pixel_values`` in older HF, ``pixel_values_videos`` in newer ones.
        params = inspect.signature(self.model.forward).parameters
        self._video_kwarg = (
            "pixel_values_videos" if "pixel_values_videos" in params else "pixel_values"
        )

        if freeze_backbone:
            # Freeze everything except the final classification head. MViT's
            # head is named ``classifier`` in HF implementations.
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

        # Gradient checkpointing: trade compute for VRAM. Useful for the Large
        # variant; usually unnecessary for Base on a 20GB GPU.
        if use_gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
            except AttributeError:
                pass

    def get_param_groups_llrd(
        self,
        base_lr: float,
        weight_decay: float,
        decay_rate: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Coarse stage-wise LR decay for MViT.

        MViT's hierarchical structure (4 stages, each with its own depth and
        feature dimensionality) doesn't map cleanly to ViT's flat block list.
        We use a **stage-level** decay instead of per-block: head gets
        ``base_lr``, stage 4 gets ``base_lr × decay``, stage 3 gets
        ``base_lr × decay^2``, etc. Embeddings get the smallest LR.

        Falls back to a single group if the param names don't expose stages.
        """
        import re

        # Try to parse stage indices from param names. HF MViT uses something
        # like "encoder.layer.{idx}." where layers are grouped by stage; the
        # mapping idx→stage is in the config. We approximate by binning by
        # layer idx into 4 quartiles.
        max_layer = -1
        for name, _ in self.model.named_parameters():
            m = re.search(r"\.layer\.(\d+)\.", name)
            if m:
                max_layer = max(max_layer, int(m.group(1)))

        if max_layer < 0:
            # Couldn't parse; fall back to a single group.
            return [
                {
                    "params": [p for p in self.model.parameters() if p.requires_grad],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            ]

        def stage_id(name: str) -> int:
            # Higher = closer to head = larger LR.
            if "classifier" in name or name.startswith("pooler"):
                return 5
            m = re.search(r"\.layer\.(\d+)\.", name)
            if m:
                # Map block idx to one of 4 stage bins.
                idx = int(m.group(1))
                return 1 + int(idx * 4 / max(max_layer + 1, 1))
            return 0  # embeddings

        groups: Dict[tuple, List[nn.Parameter]] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            sid = stage_id(name)
            no_decay = (
                param.ndim <= 1
                or name.endswith(".bias")
                or "layernorm" in name.lower()
                or "norm" in name.lower()
                or "pos_embed" in name.lower()
                or "position_embeddings" in name.lower()
            )
            groups.setdefault((sid, no_decay), []).append(param)

        param_groups: List[Dict[str, Any]] = []
        max_stage = 5
        for (sid, no_decay), params in sorted(groups.items()):
            multiplier = decay_rate ** (max_stage - sid)
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

        # Temporal: replicate to native frame count, preserving order.
        if T != self.native_frames:
            video = video.repeat_interleave(self._repeat, dim=1)

        # Spatial: resize to native image_size.
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
