"""Date-compressed and hierarchical classifiers over pre-anchor history.

Each date is an unordered event bag.  Both models share exactly the same date
embedding and mean/last/static readout.  The hierarchical variant replaces one
three-layer global encoder with one local layer plus two global layers; it is a
Hi-BEHRT-inspired local-to-global classifier, not an exact paper reproduction.
All attention is bidirectional within the supplied pre-anchor history.  These
models have no autoregressive pretraining objective.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import PAD_ID, TimelineTransformer


def _encoder(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
    encoder.apply(TimelineTransformer._initialize)
    return encoder


def _encoder_slice(source: nn.TransformerEncoder, start: int, count: int) -> nn.TransformerEncoder:
    """Copy a contiguous layer range without consuming additional random draws."""
    result = nn.TransformerEncoder(source.layers[start], num_layers=count, enable_nested_tensor=False)
    for destination, origin in zip(result.layers, source.layers[start:start + count]):
        destination.load_state_dict(origin.state_dict())
    return result


def _masked_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


def _last_state(hidden: Tensor, mask: Tensor) -> Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    indices = positions.masked_fill(~mask, -1).max(dim=1).values.clamp_min(0)
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


class _DateModel(nn.Module):
    """The common input/readout budget, initialized before either backbone."""

    def __init__(
        self, vocab_size: int, n_targets: int, n_static: int, max_events: int,
        d_model: int, n_heads: int, n_layers: int, dropout: float,
        max_unit_events: int, bag_normalization: str = "mean",
    ) -> None:
        super().__init__()
        if vocab_size < 4 or n_targets < 1 or n_static < 0 or max_events < 1:
            raise ValueError("Invalid vocabulary, target, static, or date capacity")
        if d_model < 1 or n_heads < 1 or d_model % n_heads or n_layers < 1:
            raise ValueError("d_model must be divisible by n_heads; layers must be positive")
        if max_unit_events < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid events-per-date capacity or dropout")
        if bag_normalization not in {"mean", "sqrt"}:
            raise ValueError("bag_normalization must be 'mean' or 'sqrt'")
        self.max_events, self.max_unit_events, self.n_static = max_events, max_unit_events, n_static
        self.d_model = d_model
        self.bag_normalization = bag_normalization
        # EmbeddingBag directly returns [B*U,D], avoiding a potentially huge
        # [B,U,E,D] embedding tensor. PAD never contributes to either mode.
        # The optional sqrt path uses a sum then an explicit count divisor,
        # without changing any learned parameter or initialization draw.
        self.token_embedding = nn.EmbeddingBag(
            vocab_size, d_model, mode="mean" if bag_normalization == "mean" else "sum", padding_idx=PAD_ID
        )
        self.position_embedding = nn.Embedding(max_events, d_model)
        self.time_projection = nn.Linear(2, d_model)
        self.count_projection = nn.Linear(1, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.readout = nn.Sequential(
            nn.Linear(2 * d_model + n_static, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(d_model, n_targets)
        # Identical construction and initialization of this common part makes
        # the flat/hierarchical date encoders and heads match under a fixed seed.
        self.apply(TimelineTransformer._initialize)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()

    def _validate(self, tokens: Tensor, time_days: Tensor, static: Tensor | None = None) -> None:
        if tokens.ndim != 3 or tokens.shape[0] < 1 or tokens.shape[1] < 1 or tokens.shape[2] < 1:
            raise ValueError("tokens must have shape [nonempty batch, dates, events per date]")
        if tokens.dtype != torch.long:
            raise TypeError("tokens must be torch.long")
        if tokens.shape[1] > self.max_events or tokens.shape[2] > self.max_unit_events:
            raise ValueError("Input exceeds model date/events-per-date capacity")
        if time_days.shape != tokens.shape[:2]:
            raise ValueError("time_days must have shape [batch, dates]")
        if static is not None and static.shape != (tokens.shape[0], self.n_static):
            raise ValueError(f"static must have shape [batch, {self.n_static}]")

    def embed_dates(self, tokens: Tensor, time_days: Tensor) -> tuple[Tensor, Tensor]:
        """Embed unordered date bags and return their safe attention mask."""
        self._validate(tokens, time_days)
        batch, units, events = tokens.shape
        counts = tokens.ne(PAD_ID).sum(dim=-1)
        real_mask = counts.ne(0)
        # Data supplies BOS=2 for an empty history.  This fallback also makes an
        # accidentally all-PAD row finite without using its padding timestamps.
        mask = real_mask.clone()
        empty = ~mask.any(dim=1)
        mask[:, 0] = mask[:, 0] | empty
        bags = self.token_embedding(tokens.reshape(batch * units, events)).reshape(batch, units, self.d_model)
        if self.bag_normalization == "sqrt":
            bags = bags / counts.to(bags.dtype).clamp_min(1).sqrt().unsqueeze(-1)
        days = time_days.to(dtype=bags.dtype).clamp_min(0)
        days = torch.where(real_mask, days, torch.zeros_like(days))
        time_features = torch.stack((torch.log1p(days) / 10.0,
                                     (days / 365.25).clamp_max(120) / 120.0), dim=-1)
        positions = torch.arange(units, device=tokens.device)
        hidden = (bags + self.position_embedding(positions).unsqueeze(0)
                  + self.time_projection(time_features)
                  + self.count_projection(torch.log1p(counts.to(bags.dtype)).unsqueeze(-1) / 5.0))
        return self.input_dropout(hidden), mask

    def _classify(self, hidden: Tensor, mask: Tensor, static: Tensor) -> Tensor:
        hidden = self.output_norm(hidden)
        combined = torch.cat((_masked_mean(hidden, mask), _last_state(hidden, mask),
                              static.to(hidden.dtype)), dim=-1)
        return self.classification_head(self.readout(combined))

    def next_event_loss(self, tokens: Tensor, time_days: Tensor, static: Tensor | None = None) -> Tensor:
        raise NotImplementedError("Date-compressed models are supervised history classifiers, not autoregressive models")


class CompressedTimelineTransformer(_DateModel):
    """Flat global attention over pooled date/event units (mean by default).

    The raw-event control can use one event per unit, i.e. E=1, with precisely
    the same architecture and parameter count as date-compressed inputs.
    """

    def __init__(
        self, vocab_size: int, n_targets: int, n_static: int, max_events: int,
        d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
        dropout: float = 0.15, max_unit_events: int = 128,
        window_size: int = 32, local_layers: int = 1, global_layers: int | None = None,
        window_stride: int | None = None,
        bag_normalization: str = "mean",
    ) -> None:
        super().__init__(vocab_size, n_targets, n_static, max_events, d_model,
                         n_heads, n_layers, dropout, max_unit_events, bag_normalization)
        self.backbone = _encoder(d_model, n_heads, n_layers, dropout)

    def encode(self, tokens: Tensor, time_days: Tensor) -> tuple[Tensor, Tensor]:
        hidden, mask = self.embed_dates(tokens, time_days)
        return self.backbone(hidden, src_key_padding_mask=~mask), mask

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        self._validate(tokens, time_days, static)
        if static is None:
            raise ValueError("Static features are required")
        hidden, mask = self.encode(tokens, time_days)
        return self._classify(hidden, mask, static)


class HierarchicalTimelineTransformer(_DateModel):
    """Local date windows followed by global window-context attention.

    Windows are pooled only after local contextualization. The default stride
    equals window_size (the non-overlapping control); a smaller stride adds
    local overlap without changing the parameter budget or initialization. The
    resulting global units already contain the same absolute date positions and
    known-anchor time features, so no extra global embedding table is added.
    With local_layers + global_layers == n_layers, its parameter count exactly
    equals CompressedTimelineTransformer's budget.
    """

    def __init__(
        self, vocab_size: int, n_targets: int, n_static: int, max_events: int,
        d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
        dropout: float = 0.15, max_unit_events: int = 128,
        window_size: int = 32, local_layers: int = 1, global_layers: int | None = None,
        window_stride: int | None = None,
        bag_normalization: str = "mean",
    ) -> None:
        global_layers = n_layers - local_layers if global_layers is None else global_layers
        window_stride = window_size if window_stride is None else window_stride
        if window_size < 1 or local_layers < 1 or global_layers < 1 or local_layers + global_layers != n_layers:
            raise ValueError("Positive local/global layers must sum to n_layers; window_size must be positive")
        if window_stride < 1 or window_stride > window_size:
            raise ValueError("window_stride must be between 1 and window_size")
        super().__init__(vocab_size, n_targets, n_static, max_events, d_model,
                         n_heads, n_layers, dropout, max_unit_events, bag_normalization)
        self.window_size = window_size
        self.window_stride = window_stride
        # Match every initial attention/feed-forward layer to the flat control,
        # not only the common embeddings/head.  Splitting/deep-copying a single
        # initialized stack consumes no new random draws and changes no budget.
        shared_layers = _encoder(d_model, n_heads, n_layers, dropout)
        self.local_encoder = _encoder_slice(shared_layers, 0, local_layers)
        self.global_encoder = _encoder_slice(shared_layers, local_layers, global_layers)

    def encode(self, tokens: Tensor, time_days: Tensor) -> tuple[Tensor, Tensor]:
        hidden, mask = self.embed_dates(tokens, time_days)
        batch, units, width = hidden.shape
        windows = 1 + math.ceil(max(units - self.window_size, 0) / self.window_stride)
        padded = self.window_size + (windows - 1) * self.window_stride
        positions = torch.arange(units, device=mask.device).unsqueeze(0) + 1
        lengths = positions.masked_fill(~mask, 0).max(dim=1).values
        needed_windows = 1 + torch.div(
            (lengths - self.window_size).clamp_min(0) + self.window_stride - 1,
            self.window_stride, rounding_mode="floor",
        )
        if padded != units:
            hidden = F.pad(hidden, (0, 0, 0, padded - units))
            mask = F.pad(mask, (0, padded - units), value=False)
        # unfold appends the date-window dimension after hidden width.
        blocks = hidden.unfold(1, self.window_size, self.window_stride).permute(0, 1, 3, 2)
        blocks = blocks.reshape(batch * windows, self.window_size, width)
        block_mask = mask.unfold(1, self.window_size, self.window_stride).reshape(batch * windows, self.window_size)
        # With overlap, extra right padding can otherwise create additional
        # partly valid trailing windows. Limit each patient to the exact number
        # required by its real history, so batching/padding cannot change it.
        eligible = torch.arange(windows, device=mask.device).unsqueeze(0) < needed_windows.unsqueeze(1)
        active = block_mask.any(dim=1) & eligible.reshape(-1)
        active_indices = active.nonzero(as_tuple=True)[0]
        # Never pass a fully masked window into attention: its softmax can be
        # NaN, and multiplying NaN by a zero mask would not repair the result.
        encoded = self.local_encoder(blocks[active], src_key_padding_mask=~block_mask[active])
        local_states = _masked_mean(encoded, block_mask[active])
        pooled = hidden.new_zeros(batch * windows, width).index_copy(0, active_indices, local_states)
        pooled = pooled.reshape(batch, windows, width)
        global_mask = active.reshape(batch, windows)
        return self.global_encoder(pooled, src_key_padding_mask=~global_mask), global_mask

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        self._validate(tokens, time_days, static)
        if static is None:
            raise ValueError("Static features are required")
        hidden, mask = self.encode(tokens, time_days)
        return self._classify(hidden, mask, static)
