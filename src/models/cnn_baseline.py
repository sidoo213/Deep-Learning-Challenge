"""
Track A / Track B: same architecture, ``pretrained`` flag toggles ImageNet weights.

Forward (conceptually):
    Input:  (batch, time, C, H, W)
    Reshape: (batch * time, C, H, W)             # each frame is an independent image
    Backbone: chosen CNN up to global average pool -> (batch * time, F, 1, 1)
    Flatten: (batch * time, F)
    Reshape: (batch, time, F)
    Mean over time: (batch, F)
    (optional) Dropout: (batch, F)
    Linear classifier: (batch, num_classes)

Backbone is configurable. The defaults are tuned for the challenge:

* ``resnet50`` (≈ 25M params, F=2048) — the new default. Gives the model
  enough capacity to handle 33 SSv2-style classes; from-scratch (Track A)
  trains well with the augmentation pipeline already in this repo, and
  for Track B the ``IMAGENET1K_V2`` weights are *much* stronger than the
  V1 weights ResNet-18 used (80.9% vs 76.1% top-1 on ImageNet).
* ``resnet18`` (≈ 11M params, F=512) — the original baseline. Kept as the
  Python ``__init__`` default so that old ``.pt`` files (whose saved cfg
  has no ``backbone`` key) keep loading without code changes.

Adding a backbone to the registry below is a 2-line change: list the
torchvision constructor + its preferred weights enum. All entries must
expose ``.fc`` as the classifier so the ``backbone.fc = nn.Identity()``
trick works uniformly.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models


# Map backbone name -> (constructor, weights-enum factory, feature dim).
# We delay the weights-enum lookup behind a lambda so older torchvision
# releases that don't ship V2 weights don't crash at import time.
_BACKBONE_REGISTRY: Dict[str, Tuple[Callable[..., nn.Module], Callable[[], "models.Weights | None"], int]] = {
    "resnet18": (
        models.resnet18,
        lambda: models.ResNet18_Weights.IMAGENET1K_V1,
        512,
    ),
    "resnet34": (
        models.resnet34,
        lambda: models.ResNet34_Weights.IMAGENET1K_V1,
        512,
    ),
    "resnet50": (
        models.resnet50,
        lambda: getattr(
            models.ResNet50_Weights, "IMAGENET1K_V2", models.ResNet50_Weights.IMAGENET1K_V1
        ),
        2048,
    ),
    "resnext50_32x4d": (
        models.resnext50_32x4d,
        lambda: getattr(
            models.ResNeXt50_32X4D_Weights,
            "IMAGENET1K_V2",
            models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
        ),
        2048,
    ),
    "wide_resnet50_2": (
        models.wide_resnet50_2,
        lambda: getattr(
            models.Wide_ResNet50_2_Weights,
            "IMAGENET1K_V2",
            models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        ),
        2048,
    ),
}


class CNNBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        backbone: str = "resnet18",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone {backbone!r}. Choose one of "
                f"{sorted(_BACKBONE_REGISTRY)}."
            )

        model_fn, weights_fn, feature_dim = _BACKBONE_REGISTRY[backbone]
        weights = weights_fn() if pretrained else None
        net = model_fn(weights=weights)

        # All registered backbones expose ``.fc`` as the ImageNet classifier;
        # replace it with identity so our own head sees the pre-pool features.
        net.fc = nn.Identity()

        self.backbone = net
        # ``nn.Identity`` has zero parameters, so adding it doesn't change the
        # state_dict — old checkpoints (saved before this field existed) still
        # load without strict=False.
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.backbone_name = backbone

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (batch_size, T, C, H, W) -> logits (batch_size, num_classes)."""
        batch_size, num_frames, channels, height, width = video_batch.shape

        # Merge batch and time so the CNN runs frame-wise: (B*T, C, H, W).
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # (B*T, F, 1, 1) -> (B*T, F).
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # Restore temporal structure: (B, T, F).
        sequence_features = frame_features.view(batch_size, num_frames, -1)

        # Simple temporal pooling: average over frames -> (B, F).
        pooled_features = sequence_features.mean(dim=1)

        # (B, num_classes).
        logits = self.classifier(self.dropout(pooled_features))
        return logits
