"""
Two-Stream Spatio-Temporal Transformer (Track A — trained from scratch).

Pipeline (see two_stream_transformer_design.md for the full rationale):
    Input:  (B, T, 3, 224, 224)
    RGB    stream CNN + spatial self-attention -> (B, T, C_rgb)
    Motion stream CNN + spatial self-attention -> (B, T, C_mot)
        (motion input = frame[t] - frame[t-1], zero for t=0)
    Concat -> Linear projection -> LayerNorm + Dropout -> (B, T, d_model)
    Prepend learnable [CLS] -> (B, T+1, d_model)
    + sinusoidal positional encoding
    Transformer encoder (multi-head self-attention + FFN, pre-norm)
    LayerNorm + head_dropout on [CLS] -> Linear -> (B, num_classes)

Robustness choices for from-scratch training on a small dataset:
  * LayerNorm right after `frame_proj` keeps the transformer input distribution
    stable (otherwise the random-init Conv backbones produce noisy features
    that the transformer struggles to absorb).
  * Separate `head_dropout` (0.5 default) provides strong regularization on
    the classifier — the head is the most overfit-prone part on tiny data.
  * Explicit weight init: Kaiming-normal for Conv2d (ReLU + BN networks),
    Xavier-uniform for Linear, trunc-normal(std=0.02) for the classifier and
    the [CLS] token. Avoids huge first-step gradients.

Best known config (ablations on val 6745 clips, 2026-05-24 — see
two_stream_transformer_design.md):
    base_channels=32                   (bc32 beats bc64 by ~6-8 pt; wider overfits)
    use_class_balanced_sampler=false   (nocbs beats cbs on top-1)
    head_dropout=0.5
    label_smoothing=0.1 (≈0.0; no top-1 signal)
    use_horizontal_flip=false          (SSv2 directional classes)
  -> use these defaults when emitting a two-stream training command.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    """Basic ResNet block: Conv-BN-ReLU-Conv-BN + skip, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _SpatialSelfAttention(nn.Module):
    """Self-attention over the H*W spatial positions of a (N, C, H, W) feature map."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)         # (N, H*W, C)
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out                    # residual
        return tokens.transpose(1, 2).view(N, C, H, W)


class _StreamBackbone(nn.Module):
    """Shallow ResNet-style backbone + spatial self-attention at the top.

    Output: (N, out_channels) per-frame descriptor (after global average pool).
    """

    def __init__(self, base_channels: int = 32, attn_heads: int = 4) -> None:
        super().__init__()
        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(_ResidualBlock(c, c), _ResidualBlock(c, c))
        self.layer2 = nn.Sequential(_ResidualBlock(c, c * 2, stride=2), _ResidualBlock(c * 2, c * 2))
        self.layer3 = nn.Sequential(_ResidualBlock(c * 2, c * 4, stride=2), _ResidualBlock(c * 4, c * 4))
        self.layer4 = nn.Sequential(_ResidualBlock(c * 4, c * 8, stride=2), _ResidualBlock(c * 8, c * 8))

        self.spatial_attn = _SpatialSelfAttention(c * 8, num_heads=attn_heads)
        self.out_channels = c * 8  # 256 by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spatial_attn(x)
        return x.mean(dim=(2, 3))  # global average pool -> (N, out_channels)


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TwoStreamTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,        # accepted for API compatibility; ignored (Track A)
        base_channels: int = 32,
        spatial_attn_heads: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        del pretrained  # Track A: never used

        self.rgb_stream = _StreamBackbone(base_channels=base_channels, attn_heads=spatial_attn_heads)
        self.motion_stream = _StreamBackbone(base_channels=base_channels, attn_heads=spatial_attn_heads)

        fused_dim = self.rgb_stream.out_channels + self.motion_stream.out_channels
        self.frame_proj = nn.Linear(fused_dim, d_model)
        # LayerNorm on the per-frame projected features: keeps the input
        # distribution to the transformer stable when the upstream Conv
        # backbones are random-init (Track A).
        self.frame_norm = nn.LayerNorm(d_model)
        self.frame_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_enc = _SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable when training from scratch
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        # Strong dropout right before the classifier: on a small dataset the
        # head overfits faster than the backbone.
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Explicit init suited for ReLU+BN convnets and a Transformer head.

        * Conv2d  : Kaiming-normal (fan_out, ReLU) — proper for the ResNet stem
                    and residual blocks (PyTorch's default is Kaiming-uniform
                    with `a=sqrt(5)`, which is awkward for ReLU).
        * Linear  : Xavier-uniform (works well for attention/FFN at init).
        * LayerNorm / BatchNorm : weight=1, bias=0.
        * cls_token, classifier : trunc_normal_(std=0.02) — keeps initial
                    logits small, avoiding huge first-step gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Overrides for tokens / classifier — small magnitude.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, C, H, W), normalized RGB frames.
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video.shape

        # --- RGB stream --------------------------------------------------
        rgb_frames = video.reshape(B * T, C, H, W)
        rgb_feats = self.rgb_stream(rgb_frames).view(B, T, -1)

        # --- Motion stream: temporal differences -------------------------
        motion = torch.zeros_like(video)
        motion[:, 1:] = video[:, 1:] - video[:, :-1]
        motion_frames = motion.reshape(B * T, C, H, W)
        motion_feats = self.motion_stream(motion_frames).view(B, T, -1)

        # --- Per-frame fusion --------------------------------------------
        fused = torch.cat([rgb_feats, motion_feats], dim=-1)  # (B, T, 2*c_out)
        tokens = self.frame_proj(fused)                       # (B, T, d_model)
        tokens = self.frame_norm(tokens)                      # stabilize for transformer
        tokens = self.frame_dropout(tokens)

        # --- Prepend [CLS] + positional encoding -------------------------
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)                # (B, T+1, d_model)
        seq = self.pos_enc(seq)

        # --- Temporal Transformer ----------------------------------------
        seq = self.transformer(seq)
        seq = self.out_norm(seq)

        # --- Classifier on [CLS] -----------------------------------------
        cls_out = self.head_dropout(seq[:, 0])
        return self.classifier(cls_out)
