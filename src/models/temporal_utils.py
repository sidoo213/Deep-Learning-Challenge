"""Temporal helpers shared between models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_interpolate(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Linearly interpolate a clip along the time axis.

    Args:
        video: ``(B, T, C, H, W)`` tensor of normalized frames.
        target_frames: desired output T'. If already equal to T, returns
            ``video`` unchanged.

    Returns:
        ``(B, target_frames, C, H, W)`` tensor.

    Notes:
        - Interpolation is **linear in pixel space**. It does NOT create new
          motion information; it only spreads existing frames along a different
          temporal axis so the downstream model sees more time tokens.
        - ``F.interpolate`` with ``mode="linear"`` requires the time axis to be
          the last spatial axis, so we move axes around and back.
    """
    B, T, C, H, W = video.shape
    if T == target_frames:
        return video
    if target_frames <= 0:
        raise ValueError(f"target_frames must be > 0, got {target_frames}")
    # Move T to last, keep B*C*H*W as "batch" for 1D interpolation.
    x = video.permute(0, 2, 3, 4, 1).reshape(B * C * H * W, 1, T)
    x = F.interpolate(x, size=target_frames, mode="linear", align_corners=True)
    x = x.reshape(B, C, H, W, target_frames).permute(0, 4, 1, 2, 3)
    return x.contiguous()
