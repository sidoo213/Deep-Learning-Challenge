"""
ViViT Factorized Encoder — pure-transformer video classifier (Track A, from scratch).

Reference: Arnab et al., 2021, "ViViT: A Video Vision Transformer", Model 2.

Pipeline:
    Input:  (B, T, 3, H, W)
        ↓
    Tubelet embedding Conv3D(kernel=tubelet_size × P × P, stride=same)
        → (B, T', N, D)        with T' = T / tubelet_size, N = (H/P)*(W/P)
        ↓
    [Spatial transformer]  process each of T' temporal positions independently
        Per t': prepend [CLS_s], add spatial pos embedding, run N_s blocks of MHSA+MLP
        Extract [CLS_s] → (B, T', D)
        ↓
    [Temporal transformer]  aggregate across time
        Prepend [CLS_t], add temporal pos embedding, run N_t blocks
        Extract [CLS_t] → (B, D)
        ↓
    LayerNorm → Linear → (B, num_classes)

Why this beats `two_stream_transformer` here:
    - Pure transformer (no CNN backbone), satisfies "transformer" requirement cleanly.
    - Tubelet embedding captures local spatio-temporal patterns directly in the
      input projection — no need for a separate motion-difference stream.
    - Factorized attention costs O(N²) + O(T'²) per block instead of O((N*T')²)
      for joint space-time attention. With N=196 and T'=2 that's 38K vs 154K ops.
    - More stable to train: spatial transformer learns per-frame features first,
      then the temporal transformer composes them.

Training-from-scratch knobs (modern transformer tricks):
    - Drop path (stochastic depth): residual branch dropped with probability p,
      linearly scaled through depth. Standard regularizer for deep transformers.
    - Layer scale: small learnable scale on each residual branch (init 1e-5).
      Makes deep stacks behave like shallow ones at init, much easier to train.
    - Pre-norm (norm_first): LayerNorm before MHA and MLP, stable for training.
    - Truncated normal init (std=0.02) on Linear/CLS/pos embeddings (ViT standard).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _DropPath(nn.Module):
    """Stochastic depth: drop the entire residual branch with probability p."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask (same across all tokens of a clip).
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob


class _LayerScale(nn.Module):
    """Per-channel learnable scale on a residual branch (CaiT, Touvron 2021).

    Init guide:
        - 10-12 layers (this model): 1.0 — residuals start "fully open"
        - 24 layers (ViT-Large depth): 0.1
        - 36+ layers (very deep): 1e-4 to 1e-6
    Initialising too small on a shallow stack kills the residual signal and
    the model fails to learn (CLS stays constant through the encoder).
    """

    def __init__(self, dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class _TransformerBlock(nn.Module):
    """Pre-norm encoder block with multi-head self-attention, MLP, drop path, layer scale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ls1 = _LayerScale(dim, layer_scale_init)
        self.drop_path1 = _DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.ls2 = _LayerScale(dim, layer_scale_init)
        self.drop_path2 = _DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Tubelet embedding
# ---------------------------------------------------------------------------


class _TubeletEmbedding(nn.Module):
    """3D Conv tubelet embedding: (B, T, 3, H, W) → (B, T', N, D).

    Non-overlapping tubelets of shape (tubelet_size, P, P) are projected to
    `embed_dim`-dim tokens.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int],
        tubelet_size: int,
    ) -> None:
        super().__init__()
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video (B, T, C, H, W) → (B, T', N, D) with T'=T/tubelet, N=H'·W'."""
        # Conv3d expects (B, C, T, H, W).
        x = video.permute(0, 2, 1, 3, 4)
        x = self.proj(x)                                   # (B, D, T', H', W')
        B, D, Tp, Hp, Wp = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, Tp, Hp * Wp, D)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ViViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int = 4,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 256,
        spatial_depth: int = 6,
        temporal_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1.0,
        pretrained: bool = False,  # accepted for API compatibility — Track A
    ) -> None:
        super().__init__()
        del pretrained

        if num_frames % tubelet_size != 0:
            raise ValueError(
                f"num_frames ({num_frames}) must be a multiple of tubelet_size ({tubelet_size})"
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be a multiple of patch_size ({patch_size})"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.temporal_dim = int(num_frames // tubelet_size)
        self.spatial_dim = int((image_size // patch_size) ** 2)
        self.embed_dim = int(embed_dim)

        # ---- Tubelet embedding ----
        self.tubelet_embed = _TubeletEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=(patch_size, patch_size),
            tubelet_size=tubelet_size,
        )

        # ---- Spatial transformer ----
        self.cls_token_s = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_s = nn.Parameter(
            torch.zeros(1, 1 + self.spatial_dim, embed_dim)
        )
        # Linear drop-path schedule across the whole stack (spatial + temporal).
        total_depth = spatial_depth + temporal_depth
        dpr = [float(x) for x in torch.linspace(0.0, drop_path_rate, total_depth)]

        self.spatial_blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(spatial_depth)
            ]
        )
        self.norm_s = nn.LayerNorm(embed_dim)

        # ---- Temporal transformer ----
        self.cls_token_t = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_t = nn.Parameter(
            torch.zeros(1, 1 + self.temporal_dim, embed_dim)
        )
        self.temporal_blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[spatial_depth + i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(temporal_depth)
            ]
        )
        self.norm_t = nn.LayerNorm(embed_dim)

        # ---- Classifier ----
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token_s, std=0.02)
        nn.init.trunc_normal_(self.cls_token_t, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_s, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_t, std=0.02)
        self.apply(self._init_module)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Named params to exclude from weight decay (ViT convention)."""
        return {"cls_token_s", "cls_token_t", "pos_embed_s", "pos_embed_t"}

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, C, H, W) → logits (B, num_classes)."""
        B, T, _, _, _ = video.shape
        if T != self.num_frames:
            raise ValueError(
                f"ViViT was built for num_frames={self.num_frames} but got T={T}"
            )

        # ---- Tubelet embed: (B, T', N, D) ----
        x = self.tubelet_embed(video)
        _, Tp, N, D = x.shape

        # ---- Spatial transformer: fold T' into batch ----
        x_s = x.reshape(B * Tp, N, D)
        cls_s = self.cls_token_s.expand(B * Tp, -1, -1)
        x_s = torch.cat([cls_s, x_s], dim=1)                 # (B·T', N+1, D)
        x_s = x_s + self.pos_embed_s
        for block in self.spatial_blocks:
            x_s = block(x_s)
        x_s = self.norm_s(x_s)
        # Per-tube spatial CLS → (B, T', D)
        cls_per_tube = x_s[:, 0].reshape(B, Tp, D)

        # ---- Temporal transformer ----
        cls_t = self.cls_token_t.expand(B, -1, -1)
        x_t = torch.cat([cls_t, cls_per_tube], dim=1)        # (B, T'+1, D)
        x_t = x_t + self.pos_embed_t
        for block in self.temporal_blocks:
            x_t = block(x_t)
        x_t = self.norm_t(x_t)

        return self.head(x_t[:, 0])
