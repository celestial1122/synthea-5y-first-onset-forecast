"""Small, explicitly controlled alternatives to the initial timeline readout.

Summary models consume ``static`` as a train-fitted full-history feature vector.
They intentionally ignore token content and timestamps, providing an efficient
comparison for the information lost by truncating the timeline. The pooled
Transformer keeps the original causal backbone and pretraining objective; only
the supervised readout changes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .models import BOS_ID, PAD_ID, TimelineTransformer, _validate_shapes


def _validate_summary_constructor(
    vocab_size: int, n_targets: int, n_static: int, max_events: int, d_model: int
) -> None:
    if vocab_size < 4 or n_targets < 1 or n_static < 1 or max_events < 1 or d_model < 1:
        raise ValueError("Summary models require a vocabulary, targets, and nonempty summary features")


class SummaryMLP(nn.Module):
    """A regularized nonlinear classifier over frozen full-history summaries.

    Layer normalization is applied to hidden states, rather than the input, so
    the supplied train-standardized clinical features retain their magnitudes.
    The narrow second hidden layer limits capacity on the small patient cohort.
    Attention arguments are accepted only for a shared experiment interface.
    """

    def __init__(
        self,
        vocab_size: int,
        n_targets: int,
        n_static: int,
        max_events: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        _validate_summary_constructor(vocab_size, n_targets, n_static, max_events, d_model)
        self.max_events = max_events
        self.n_static = n_static
        bottleneck = max(8, d_model // 2)
        self.summary_encoder = nn.Sequential(
            nn.Linear(n_static, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(bottleneck, n_targets)
        self.apply(TimelineTransformer._initialize)

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        if static is None:
            raise ValueError("Summary models require static summary features")
        _validate_shapes(tokens, time_days, static, self.max_events, self.n_static)
        summary = static.to(dtype=self.classification_head.weight.dtype)
        return self.classification_head(self.summary_encoder(summary))

    def next_event_loss(
        self, tokens: Tensor, time_days: Tensor, static: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError("A summary classifier does not have a next-event objective")


class SummaryLinear(nn.Module):
    """Pure logistic baseline over the same frozen full-history features."""

    def __init__(
        self,
        vocab_size: int,
        n_targets: int,
        n_static: int,
        max_events: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        _validate_summary_constructor(vocab_size, n_targets, n_static, max_events, d_model)
        self.max_events = max_events
        self.n_static = n_static
        self.classification_head = nn.Linear(n_static, n_targets)
        self.apply(TimelineTransformer._initialize)

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        if static is None:
            raise ValueError("Summary models require static summary features")
        _validate_shapes(tokens, time_days, static, self.max_events, self.n_static)
        return self.classification_head(static.to(dtype=self.classification_head.weight.dtype))

    def next_event_loss(
        self, tokens: Tensor, time_days: Tensor, static: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError("A summary classifier does not have a next-event objective")


class PooledTimelineTransformer(TimelineTransformer):
    """Original causal timeline encoder with a richer supervised readout.

    Mean pooling excludes both PAD and BOS, and the last valid state preserves
    the original recent-context readout. A small nonlinear fusion layer lets
    static age/sex (or supplied summaries) interact with those history states.
    The inherited encode/next_event_loss implementations remain unchanged.
    """

    def __init__(
        self,
        vocab_size: int,
        n_targets: int,
        n_static: int,
        max_events: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            n_targets=n_targets,
            n_static=n_static,
            max_events=max_events,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.pooling_fusion = nn.Sequential(
            nn.Linear(2 * d_model + n_static, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(d_model, n_targets)
        self.pooling_fusion.apply(self._initialize)
        self._initialize(self.classification_head)

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        _validate_shapes(tokens, time_days, static, self.max_events, self.n_static)
        hidden = self.encode(tokens, time_days)
        event_mask = tokens.ne(PAD_ID) & tokens.ne(BOS_ID)
        event_count = event_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean_state = (hidden * event_mask.unsqueeze(-1)).sum(dim=1) / event_count
        last_positions = tokens.ne(PAD_ID).sum(dim=1).sub(1).clamp_min(0)
        last_state = hidden[
            torch.arange(tokens.shape[0], device=tokens.device), last_positions
        ]
        combined = torch.cat((mean_state, last_state, static.to(hidden.dtype)), dim=-1)
        return self.classification_head(self.pooling_fusion(combined))
