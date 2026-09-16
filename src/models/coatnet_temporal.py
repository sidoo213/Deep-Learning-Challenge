"""
CoAtNet (Dai et al., NeurIPS 2021) used as a per-frame encoder, followed
by temporal mean pooling and a linear head.

Architectural recap (CoAtNet):
    S0 (stem)   conv 3x3
    S1, S2      MBConv blocks      (pure convolution, locality bias)
    S3, S4      Transformer blocks (RELATIVE attention)
    Pool        global average
    Head        Linear

Why CoAtNet for this challenge:
    * The 5-stage "conv first, attention later" structure was designed
      specifically to train from random init on small-to-medium datasets
      (Dai et al. show CoAtNet beating ViT at every scale, especially the
      smallest). Our ~30k clips sit comfortably in the regime where the
      MBConv stages give critical translation/locality prior, while the
      relative-attention stages still let the model build long-range
      relations between the hand-object pair across the image.
    * Architecturally complementary to MaxViT (which mixes attention and
      conv WITHIN every block). For an ensemble, CoAtNet errors are
      partially decorrelated from MaxViT's — typically +0.3-0.7 top-1 in
      the final mix.

Backbone source:
    timm's `coatnet_*_rw_224` family. These are Ross Wightman's tuned
    variants (better-than-paper from-scratch numbers on ImageNet-1k). The
    `_rw_` postfix marks the tuned versions; `_rmlp_` adds a relative MLP
    head (slightly stronger but heavier).

Default variant:
    `coatnet_0_rw_224` — ~25M params, drop-in match to MaxViT-Tiny.

Pipeline:
    Input:                    (B, T, C, H, W) with H=W=224, C=3
    Reshape:                  (B*T, C, H, W)
    timm CoAtNet forward:     (B*T, feature_dim)
        (num_classes=0 + global_pool='avg' tells timm to drop its own
         head AND apply global average pool → we get the pooled per-frame
         embedding directly, no manual unpooling.)
    Reshape + temporal mean:  (B, feature_dim)
    Dropout + Linear:         (B, num_classes)

Efficiency knobs:
    * variant — pick the size that matches your VRAM:
        coatnet_0_rw_224     ~25M params   (default; matches MaxViT-Tiny)
        coatnet_1_rw_224     ~42M params
        coatnet_2_rw_224     ~75M params
        coatnet_3_rw_224     ~166M params
        coatnet_nano_rw_224  ~15M params   (faster iteration / sanity check)
      Any other timm CoAtNet name is passed through without validation.
    * use_gradient_checkpointing — trades ~30% wall-clock for 30-50% VRAM.
      Useful for coatnet_1+ on tight memory; off by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoAtNetTemporal(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        variant: str = "coatnet_0_rw_224",
        dropout: float = 0.3,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # Lazy import so the rest of the project keeps loading without
        # timm. Surfaces a clear error message only when the user picks
        # this model.
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "coatnet_temporal requires the `timm` package. "
                "Run `uv sync` (it is declared in pyproject.toml) or "
                "`pip install timm`."
            ) from e

        # num_classes=0 + global_pool='avg' tells timm to:
        #   - drop the classification head (replace with Identity)
        #   - keep the global average pool
        # so model(x) returns the pooled (B*T, F) embedding directly.
        self.backbone = timm.create_model(
            variant,
            pretrained=bool(pretrained),
            num_classes=0,
            global_pool="avg",
        )
        # timm exposes the pooled feature dim through ``num_features``.
        self.feature_dim = int(self.backbone.num_features)

        if use_gradient_checkpointing and hasattr(
            self.backbone, "set_grad_checkpointing"
        ):
            self.backbone.set_grad_checkpointing(enable=True)

        # nn.Identity / nn.Dropout carry zero parameters, so legacy ``.pt``
        # files (saved before this dropout knob existed) load without
        # state_dict mismatches if you ever swap dropout on/off.
        self.dropout = (
            nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (B, T, C, H, W) -> logits (B, num_classes)."""
        batch_size, num_frames, channels, height, width = video_batch.shape

        # Merge batch and time: each frame is independently encoded.
        frames = video_batch.reshape(
            batch_size * num_frames, channels, height, width
        )
        # (B*T, feature_dim) thanks to num_classes=0 + global_pool='avg'.
        frame_features = self.backbone(frames)

        # Restore temporal structure, then collapse time by mean pooling.
        sequence_features = frame_features.view(
            batch_size, num_frames, self.feature_dim
        )
        pooled = sequence_features.mean(dim=1)

        return self.classifier(self.dropout(pooled))
