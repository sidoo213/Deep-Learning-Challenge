"""
Temporal Shift Module (TSM) — Track A, trained from scratch.

Reference: Lin et al., "TSM: Temporal Shift Module for Efficient Video
Understanding" (ICCV 2019). The key idea: inside a 2D ResNet, shift a
small fraction of the channels along the temporal axis. This gives the
2D CNN temporal reasoning ability for **zero extra parameters** and
negligible FLOPs.

Pipeline:
    Input:  (B, T, 3, 224, 224)
    Reshape -> (B*T, 3, 224, 224)
    ResNet-style stages, each residual block prefixed with a temporal shift
    Global average pool -> (B*T, C)
    Reshape -> (B, T, C)
    Mean pool over T -> (B, C)
    Linear -> (B, num_classes)

Best known config (ablations on val 6745 clips, 2026-05-24 — see tsm.md §11):
    optimizer=adamw, lr=1e-3
    scheduler=cosine, warmup_epochs=3
    use_class_balanced_sampler=false   (nocbs beats cbs on top-1)
    temporal_pool=mean                 (mean beats the learned tconv)
    dropout=0.5
    label_smoothing=0.1 (≈0.0; no top-1 signal)
    use_horizontal_flip=false          (SSv2 directional classes)
  -> use these defaults when emitting a TSM training command.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _temporal_shift(x: torch.Tensor, num_segments: int, fold_div: int = 8) -> torch.Tensor:
    """Shift `1/fold_div` of channels forward in time, another `1/fold_div`
    backward in time, leave the rest untouched.

    x: (B*T, C, H, W)
    """
    nt, c, h, w = x.shape
    b = nt // num_segments
    x = x.view(b, num_segments, c, h, w)

    fold = c // fold_div
    # Allocate uninitialized memory and write every element exactly once,
    # instead of zeroing the whole tensor and then overwriting ~75% of it.
    # The only cells that must be explicitly zeroed are the two temporal
    # boundaries that shift in from "outside" the clip.
    out = torch.empty_like(x)
    # leave the rest untouched (channels 2*fold: are copied as-is)
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    # shift left  (frame t gets channels from frame t+1); last frame -> zero
    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, -1, :fold] = 0.0
    # shift right (frame t gets channels from frame t-1); first frame -> zero
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
    out[:, 0, fold:2 * fold] = 0.0

    return out.view(nt, c, h, w)


class _TSMResidualBlock(nn.Module):
    """ResNet basic block with a temporal shift applied to the first conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_segments: int,
        stride: int = 1,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

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
        # x: (B*T, C, H, W)
        identity = self.shortcut(x)
        shifted = _temporal_shift(x, self.num_segments, self.fold_div)  # the magic
        out = self.relu(self.bn1(self.conv1(shifted)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _MeanTemporalPool(nn.Module):
    """Average over the T axis (original TSM/TSN consensus). Zero params, so
    checkpoints trained before `temporal_pool` existed load unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, F)
        return x.mean(dim=1)


class _AttentionTemporalPool(nn.Module):
    """Single-head additive attention over T: learns a per-frame relevance
    weight and returns a weighted sum. Strictly more expressive than the mean
    when some frames carry more signal than others.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, F)
        weights = torch.softmax(self.score(torch.tanh(self.proj(x))), dim=1)  # (B,T,1)
        return (x * weights).sum(dim=1)


class _TConvTemporalPool(nn.Module):
    """Depthwise 1-D convolution over the T axis, then mean pool.

    Unlike mean/attention pooling, a temporal conv is **order-sensitive** (its
    kernel sees the frames in sequence), which is the right inductive bias for
    an anticipation task. Depthwise (`groups=dim`) keeps it to ~3*dim params.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.tconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
            groups=dim, bias=False,
        )
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, F)
        y = x.transpose(1, 2)                 # (B, F, T)
        y = self.act(self.bn(self.tconv(y)))  # (B, F, T)
        return y.transpose(1, 2).mean(dim=1)  # (B, F)


class TSM(nn.Module):
    """ResNet18-style backbone with Temporal Shift Modules + temporal pooling."""

    def __init__(
        self,
        num_classes: int,
        num_segments: int = 4,
        pretrained: bool = False,  # accepted for API compatibility — Track A ignores it
        base_channels: int = 64,
        dropout: float = 0.5,
        fold_div: int = 8,
        temporal_pool: str = "mean",
        resnet34: bool = False,
    ) -> None:
        super().__init__()
        del pretrained
        self.num_segments = num_segments

        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def make_layer(in_c: int, out_c: int, blocks: int, stride: int) -> nn.Sequential:
            layers = [_TSMResidualBlock(in_c, out_c, num_segments, stride=stride, fold_div=fold_div)]
            for _ in range(1, blocks):
                layers.append(_TSMResidualBlock(out_c, out_c, num_segments, stride=1, fold_div=fold_div))
            return nn.Sequential(*layers)

        # Depth: ResNet-18 layout (2,2,2,2) by default; ResNet-34 (3,4,6,3)
        # when resnet34=True. Both use BasicBlocks (Bottleneck would be R-50+).
        b1, b2, b3, b4 = (3, 4, 6, 3) if resnet34 else (2, 2, 2, 2)
        self.layer1 = make_layer(c, c, blocks=b1, stride=1)
        self.layer2 = make_layer(c, c * 2, blocks=b2, stride=2)
        self.layer3 = make_layer(c * 2, c * 4, blocks=b3, stride=2)
        self.layer4 = make_layer(c * 4, c * 8, blocks=b4, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        feature_dim = c * 8
        if temporal_pool == "mean":
            self.temporal_pool: nn.Module = _MeanTemporalPool()
        elif temporal_pool == "attention":
            self.temporal_pool = _AttentionTemporalPool(feature_dim)
        elif temporal_pool == "tconv":
            self.temporal_pool = _TConvTemporalPool(feature_dim)
        else:
            raise ValueError(
                f"Unknown temporal_pool={temporal_pool!r} "
                "(use 'mean', 'attention', or 'tconv')."
            )
        self.temporal_pool_kind = temporal_pool

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, C, H, W)   with T == self.num_segments
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video.shape
        assert T == self.num_segments, (
            f"TSM was built for num_segments={self.num_segments} but got T={T}"
        )

        x = video.reshape(B * T, C, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)                   # (B*T, C_out)
        x = x.view(B, T, -1)                         # (B, T, C_out)
        x = self.temporal_pool(x)                    # temporal aggregation -> (B, C_out)
        x = self.dropout(x)
        return self.classifier(x)
