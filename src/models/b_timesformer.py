"""
TimeSformer wrapper for Track B (open world).

Loads ``facebook/timesformer-base-finetuned-ssv2`` and swaps its 174-class head
to the challenge's ``num_classes`` (33).

Why TimeSformer for this challenge:
    First transformer model designed specifically for video (Facebook AI, 2021).
    Uses "divided space-time attention": each block does spatial attention then
    temporal attention. Its base variant on Hugging Face is fine-tuned on the
    Something-Something V2 task — same data family as our challenge.

Notable difference vs VideoMAE:
    TimeSformer's native ``num_frames`` is 8 and VideoMAE's is 16. The project's
    ``processed_data`` only ships 4 frames per clip, so the wrapper interpolates
    the pretrained temporal embeddings down to ``num_frames=4`` (see below).

Input / output:
    forward(video) takes ``(B, T, C, H, W)`` with ImageNet-normalized RGB frames
    and returns logits of shape ``(B, num_classes)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_id: str = "facebook/timesformer-base-finetuned-ssv2",
        num_frames: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        from transformers import TimesformerConfig, TimesformerForVideoClassification

        if pretrained:
            # Load with the model's native num_frames (no override) so the
            # pretrained time_embeddings (shape (1, native_frames, hidden)) are
            # loaded intact. We then interpolate them to our num_frames below.
            self.model = TimesformerForVideoClassification.from_pretrained(
                model_id,
                num_labels=int(num_classes),
                ignore_mismatched_sizes=True,
            )

            native_frames = int(self.model.config.num_frames)
            target_frames = int(num_frames)
            if target_frames != native_frames:
                # Linearly interpolate the learned temporal embeddings from
                # native_frames positions down to target_frames positions.
                # Without this, HF reinits them randomly and the temporal
                # attention is effectively destroyed — training stalls near
                # uniform predictions at our base LR.
                old = self.model.timesformer.embeddings.time_embeddings.data
                # (1, native_frames, hidden) -> (1, hidden, native_frames)
                resized = F.interpolate(
                    old.transpose(1, 2),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
                self.model.timesformer.embeddings.time_embeddings = nn.Parameter(
                    resized.contiguous()
                )
                self.model.config.num_frames = target_frames
        else:
            config = TimesformerConfig(
                num_labels=int(num_classes), num_frames=int(num_frames)
            )
            self.model = TimesformerForVideoClassification(config)

        # Native spatial size; default 224 on the published checkpoints.
        self.image_size = int(getattr(self.model.config, "image_size", 224))

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, C, H, W) -> logits (B, num_classes)."""
        B, T, C, H, W = video.shape

        # Spatial adaptation: bilinear resize to the native input size if
        # the dataloader produces a different one. Mirrors the pattern used
        # by b_vjepa2 / b_mvitv2 / b_videomae.
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
