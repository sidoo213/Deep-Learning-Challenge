"""
VideoMAE wrapper for Track B (open world).

Loads ``MCG-NJU/videomae-base-finetuned-ssv2`` (or any compatible checkpoint),
swaps the classification head to match the challenge's ``num_classes`` (33),
and exposes the standard ``(B, T, C, H, W) -> (B, num_classes)`` interface used
everywhere else in this repo.

Why VideoMAE for this challenge:
    The pretrained checkpoint was fine-tuned on the full Something-Something V2
    dataset (174 classes). Our challenge is a 33-class subset of the same data
    distribution, so the temporal/spatial features are already aligned with the
    target task — we mostly need to relearn the final linear layer.

Input / output:
    forward(video) takes ``(B, T, C, H, W)`` with ImageNet-normalized RGB frames
    (the existing ``build_transforms(use_imagenet_norm=True)`` produces exactly
    this) and returns logits of shape ``(B, num_classes)``.

Note on num_frames:
    The HF VideoMAE checkpoints have temporal positional embeddings sized for a
    fixed ``num_frames`` (16 for the base/short variants on SSv2). Passing a
    different number of frames triggers a config mismatch unless we explicitly
    set ``num_frames`` at load time. We let the wrapper do this automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoMAEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_id: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frames: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Local import so the rest of the codebase keeps working without
        # transformers installed (Track A users don't need this dependency).
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification

        if pretrained:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                model_id,
                num_labels=int(num_classes),
                num_frames=int(num_frames),
                ignore_mismatched_sizes=True,
            )
        else:
            # Same architecture but trained from scratch. Mostly here for ablation
            # — Track B normally uses pretrained=True.
            config = VideoMAEConfig(
                num_labels=int(num_classes), num_frames=int(num_frames)
            )
            self.model = VideoMAEForVideoClassification(config)

        # Native spatial input size; default 224 on all the published
        # checkpoints. Stored so ``forward`` can bilinearly resize inputs
        # that don't match — same pattern as the other ``b_*`` wrappers.
        self.image_size = int(getattr(self.model.config, "image_size", 224))

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, C, H, W) -> logits (B, num_classes)."""
        B, T, C, H, W = video.shape

        # Spatial adaptation: bilinear resize to the model's native size if
        # the dataloader produces a different one. Without this, a config
        # mismatch crashes deep inside the patch-embedding conv.
        if H != self.image_size or W != self.image_size:
            flat = video.reshape(B * T, C, H, W)
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            video = flat.reshape(B, T, C, self.image_size, self.image_size)

        outputs = self.model(pixel_values=video)
        return outputs.logits
