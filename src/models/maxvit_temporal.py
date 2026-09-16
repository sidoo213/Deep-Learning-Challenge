"""
MaxViT-Tiny used as a per-frame encoder, followed by temporal mean pooling
and a linear head.

Reference: Tu et al., "MaxViT: Multi-Axis Vision Transformer", ECCV 2022.

Why MaxViT for this challenge:
    Each MaxViT block stacks three primitives:
        1. MBConv (depthwise + squeeze-and-excitation) — classic CNN locality.
        2. Block attention   — self-attention inside non-overlapping 7×7
                                windows. This is the LOCAL attention the
                                wrapper name advertises.
        3. Grid attention    — self-attention on a sparse, dilated grid that
                                covers the whole image. Sparse-global mixer.
    The combination gives a strong CNN-style inductive bias (translation
    equivariance, locality) plus a cheap O(N) global receptive field. This
    profile is closer to ResNets than to plain ViT, which makes MaxViT one
    of the few attention-based backbones that trains well from scratch on
    small datasets (~30k clips here).

    The wrapper bolts MaxViT-Tiny on top of the standard
    ``(B, T, C, H, W) → (B, num_classes)`` interface used everywhere else
    in this repo. Temporal aggregation is a simple mean over T, mirroring
    ``cnn_baseline``: with T=4 there is not enough signal to justify a
    learnt temporal head on top of a much heavier spatial encoder.

Pipeline:
    Input:                (B, T, C, H, W) with C=3, H=W=224
    Reshape:              (B*T, C, H, W)
    MaxViT-Tiny stem:     (B*T, 64, H/2, W/2)
    MaxViT-Tiny blocks:   (B*T, 512, H/32, W/32)
    Global avg pool:      (B*T, 512)
    Temporal mean over T: (B,   512)
    Dropout + Linear:     (B,   num_classes)

Notes:
    * MaxViT-Tiny is ~31M parameters — heavier than ResNet-18/50 but lighter
      than the two-stream + transformer model. On a 24 GB GPU at 224×224 with
      T=4 frames and AMP, batch_size ≈ 16 is the safe operating point.
    * The torchvision implementation hard-codes the partition-window size
      (7×7), so the input MUST be 224×224. Override
      ``dataset.image_size=224`` if your default differs.
    * For Track A (closed world), instantiate with ``pretrained=False`` so
      the backbone trains from random init. For Track B you can flip the
      flag — torchvision ships ImageNet-1k V1 weights — but the gain is
      modest because the pretraining domain is image classification, not
      video action anticipation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaxViTTemporal(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Lazy import so the project still loads if torchvision is older
        # than 0.16 (when maxvit_t landed). Surfaces a clean error message
        # if the user picks this model on a stale install.
        try:
            from torchvision.models import maxvit_t, MaxVit_T_Weights
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "maxvit_temporal requires torchvision >= 0.16 (for maxvit_t). "
                "Upgrade torchvision and try again."
            ) from e

        weights = MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = maxvit_t(weights=weights)

        # Discover the feature dim by inspecting the original classifier's
        # last Linear layer. For MaxViT-Tiny this is 512.
        last_linear: nn.Linear | None = None
        for module in self.backbone.classifier.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise RuntimeError(
                "Could not locate a Linear inside maxvit_t.classifier; "
                "torchvision API may have changed."
            )
        self.feature_dim = int(last_linear.in_features)

        # Run features manually (we skip the original classifier entirely),
        # then apply our own dropout + head sized for the challenge.
        # nn.Identity / nn.Dropout carry zero parameters, so swapping them
        # in/out leaves the state_dict layout unchanged.
        self.dropout = (
            nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run a (B*T, C, H, W) batch through MaxViT-Tiny's stem + blocks,
        then global-average-pool to (B*T, feature_dim)."""
        # torchvision's MaxViT stores blocks as a ModuleList, not Sequential,
        # so iterate explicitly.
        x = self.backbone.stem(x)
        for block in self.backbone.blocks:
            x = block(x)
        # x: (B*T, F, H', W') — pool across the remaining spatial axes.
        return x.mean(dim=(-2, -1))

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (B, T, C, H, W) -> logits (B, num_classes)."""
        batch_size, num_frames, channels, height, width = video_batch.shape

        # torchvision's maxvit_t hard-codes a 7x7 partition window. The final
        # feature map at 224 input is 14x14, which tiles cleanly into 2x2
        # 7x7 windows. Any other H/W would produce a non-tiling grid and the
        # block-attention reshape crashes with an opaque error. Catch it here.
        if height != 224 or width != 224:
            raise ValueError(
                f"maxvit_temporal requires 224x224 input (torchvision's "
                f"maxvit_t partitions in 7x7 windows that only tile cleanly "
                f"at this size). Got {height}x{width}. Set "
                "`dataset.image_size=224` to fix."
            )

        # Run all frames through the same spatial encoder.
        frames = video_batch.reshape(
            batch_size * num_frames, channels, height, width
        )
        frame_features = self._frame_features(frames)  # (B*T, F)

        # Restore temporal structure, then collapse time by mean pooling.
        sequence_features = frame_features.view(
            batch_size, num_frames, self.feature_dim
        )
        pooled = sequence_features.mean(dim=1)  # (B, F)

        return self.classifier(self.dropout(pooled))
