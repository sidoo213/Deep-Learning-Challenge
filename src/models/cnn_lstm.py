"""
CNN + Bidirectional LSTM with attention pooling 
Forward:
    (B, T, C, H, W)
      ─► ResNet18 per frame ─► (B*T, 512)
      ─► LayerNorm + Dropout ─► (B, T, 512)
      ─► 2-layer bidirectional LSTM ─► (B, T, 2*hidden)
      ─► Additive attention pool over time ─► (B, 2*hidden)
      ─► LayerNorm + Dropout + Linear ─► (B, num_classes)

Why each piece (versus the naive "ResNet18 → 1-layer LSTM → last hidden → Linear"):

  * **Bidirectional**: each frame sees both past *and* future context within the
    short clip. For anticipation, "what's coming next?" benefits enormously from
    knowing what just happened *and* what's framed at the very end.

  * **Attention pooling over T frames** (not last_hidden only): with T=4, taking
    only the last LSTM step throws away 75% of the temporal information. Additive
    attention learns a per-frame relevance weight and produces a weighted sum.

  * **2 stacked LSTM layers with inter-layer dropout (0.3)**: deeper temporal
    reasoning; the dropout is mandatory because the backbone is trained from
    scratch on a small dataset (Track A) and overfits aggressively otherwise.

  * **LayerNorm on per-frame features**: stabilizes the unbounded ResNet18 output
    before the LSTM, which would otherwise see exploding hidden states early in
    training (no pretraining ⇒ feature distribution is noisy at init).

  * **Strong head dropout (0.5) + LayerNorm**: the head is the most overfit-prone
    part with so few samples per class.

  * **Careful LSTM init**: orthogonal recurrent weights + forget-gate bias = 1
    (Jozefowicz et al. 2015) — well-known trick that significantly speeds up
    LSTM training, especially when the upstream features are random-init noisy.

  * **Truncated-normal classifier init (std=0.02)**: keeps initial logits small,
    avoiding huge first-step gradients when the entire network is random-init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _init_lstm_(lstm: nn.LSTM) -> None:
    """Orthogonal init for recurrent weights, Xavier for input weights, zero
    biases except the forget gate (set to 1 so the cell remembers by default).

    Reference: Jozefowicz, R. et al. "An empirical exploration of recurrent
    network architectures." ICML 2015.
    """
    for name, p in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias" in name:
            nn.init.zeros_(p.data)
            # PyTorch packs the four LSTM gates' biases into one tensor of
            # length 4*hidden in order [input, forget, cell, output]. We
            # selectively set the forget-gate slice to 1.
            n = p.size(0)
            p.data[n // 4 : n // 2].fill_(1.0)


class AttentionPool(nn.Module):
    """Single-head additive attention pooling over a temporal axis.

    Equivalent to::

        scores  = v^T · tanh(W · h_t)        # (B, T, 1)
        weights = softmax_t(scores)          # (B, T, 1)
        output  = Σ_t weights_t · h_t        # (B, H)

    Cheap (two small Linears, no softmax over a long sequence) and well-suited
    to T ∈ [4, 16] frames. Strictly better than mean pool when some frames
    are more discriminative than others.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = self.score(torch.tanh(self.proj(x)))   # (B, T, 1)
        weights = F.softmax(scores, dim=1)              # (B, T, 1)
        return (x * weights).sum(dim=1)                 # (B, H)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        feature_dropout: float = 0.2,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Backbone: ResNet18. In Track A `pretrained=False` (closed world);
        # nothing in this module relies on pretrained features, but we expose
        # the flag so the same class is reusable in Track B if desired.
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Per-frame feature regularization. Random-init ResNet18 produces noisy
        # 512-d vectors that would dominate the LSTM's hidden state if not
        # normalized — LayerNorm + small dropout keeps things stable.
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)

        # Bidirectional, 2-layer LSTM. With hidden=256 and bidirectional=True
        # the output dim is 512, matching the feature_dim (no extra projection).
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        _init_lstm_(self.lstm)

        pooled_dim = 2 * lstm_hidden_size  # bidirectional concatenation

        # Temporal pooling: attention over T frames (uses every frame, not just
        # the last LSTM step).
        self.pool = AttentionPool(pooled_dim)

        # Classification head with normalization + heavy dropout.
        self.post_pool_norm = nn.LayerNorm(pooled_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(pooled_dim, num_classes)

        # Small-magnitude classifier init: prevents huge gradients in the
        # first optimizer step when the whole network is random-initialized.
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (B, T, C, H, W) -> logits (B, num_classes)."""
        B, T, C, H, W = video_batch.shape

        # Per-frame CNN encode. Collapse time into batch for a single GPU call,
        # then unfold back.
        frames = video_batch.reshape(B * T, C, H, W)
        feats = self.backbone(frames)
        feats = torch.flatten(feats, start_dim=1)        # (B*T, 512)
        feats = feats.view(B, T, -1)                     # (B, T, 512)

        feats = self.feature_norm(feats)
        feats = self.feature_dropout(feats)

        # Temporal modeling. seq: (B, T, 2*hidden).
        seq, _ = self.lstm(feats)

        # Attention pool: collapse the T axis with learned per-frame weights.
        pooled = self.pool(seq)                          # (B, 2*hidden)

        pooled = self.post_pool_norm(pooled)
        pooled = self.head_dropout(pooled)
        return self.classifier(pooled)
