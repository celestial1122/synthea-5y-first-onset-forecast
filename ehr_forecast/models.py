"""Small PyTorch models trained exclusively on the supplied patient histories.

Inputs use right padding (PAD=0) and always begin with BOS=2.  The optional
next-event task predicts token t+1 from tokens <=t; future event timestamps are
never given to the prediction at t.  MASK=3 is reserved by preprocessing but is
not used by this autoregressive objective.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PAD_ID = 0
BOS_ID = 2


def _validate_shapes(
    tokens: Tensor,
    time_days: Tensor,
    static: Tensor | None,
    max_events: int,
    n_static: int,
) -> None:
    """Validate metadata without synchronizing a GPU for per-value checks."""
    if tokens.ndim != 2 or tokens.shape[1] < 1:
        raise ValueError("tokens must have shape [batch, nonempty sequence]")
    if tokens.shape[1] > max_events:
        raise ValueError(f"Sequence length exceeds max_events={max_events}")
    if tokens.dtype != torch.long:
        raise TypeError("tokens must be torch.long")
    if time_days.shape != tokens.shape:
        raise ValueError("time_days must have the same shape as tokens")
    if static is not None and static.shape != (tokens.shape[0], n_static):
        raise ValueError(f"static must have shape [batch, {n_static}]")


class TimelineTransformer(nn.Module):
    """Decoder-only event Transformer with a patient-level multilabel head.

    Absolute sequence positions preserve event order.  Two bounded/scaled
    transformations of days before the known prediction anchor represent time.
    Static features are joined only at the classification head, so event
    pretraining shares the complete timeline backbone without a separate model.
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
        super().__init__()
        if vocab_size < 4 or n_targets < 1 or n_static < 0 or max_events < 1:
            raise ValueError("Invalid vocabulary, target, static, or sequence size")
        if d_model < 1 or n_heads < 1 or d_model % n_heads or n_layers < 1:
            raise ValueError("d_model must be divisible by n_heads; layers must be positive")
        self.max_events = max_events
        self.n_static = n_static
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_events, d_model)
        self.time_projection = nn.Linear(2, d_model)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors do not benefit these small, fixed padded batches and
        # have historically imposed additional mask/norm restrictions.
        self.backbone = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.classification_dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(d_model + n_static, n_targets)
        self.language_head = nn.Linear(d_model, vocab_size, bias=False)
        # Initialize cloned encoder layers independently rather than starting
        # all layers from the identical TransformerEncoder template weights.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def encode(self, tokens: Tensor, time_days: Tensor) -> Tensor:
        """Return causal hidden states [batch, length, d_model]."""
        _validate_shapes(tokens, time_days, None, self.max_events, self.n_static)
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        days = time_days.to(dtype=self.token_embedding.weight.dtype).clamp_min(0)
        time_features = torch.stack(
            (torch.log1p(days) / 10.0, (days / 365.25).clamp_max(120) / 120.0),
            dim=-1,
        )
        hidden = (
            self.token_embedding(tokens)
            + self.position_embedding(positions).unsqueeze(0)
            + self.time_projection(time_features)
        )
        causal_mask = torch.ones(
            length, length, device=tokens.device, dtype=torch.bool
        ).triu(diagonal=1)
        hidden = self.backbone(
            self.input_dropout(hidden),
            mask=causal_mask,
            src_key_padding_mask=tokens.eq(PAD_ID),
            is_causal=True,
        )
        return self.output_norm(hidden)

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        """Return uncalibrated disease logits, never hard labels."""
        _validate_shapes(tokens, time_days, static, self.max_events, self.n_static)
        hidden = self.encode(tokens, time_days)
        last_positions = tokens.ne(PAD_ID).sum(dim=1).sub(1).clamp_min(0)
        patient_state = hidden[
            torch.arange(tokens.shape[0], device=tokens.device), last_positions
        ]
        combined = torch.cat((patient_state, static.to(patient_state.dtype)), dim=-1)
        return self.classification_head(self.classification_dropout(combined))

    def next_event_loss(
        self, tokens: Tensor, time_days: Tensor, static: Tensor | None = None
    ) -> Tensor:
        """Mean cross-entropy over observed next tokens, ignoring PAD targets.

        A BOS-only patient contributes no target.  The zero-target batch case
        returns a differentiable zero instead of cross_entropy's empty mean.
        """
        _validate_shapes(tokens, time_days, static, self.max_events, self.n_static)
        if tokens.shape[1] == 1:
            return self.token_embedding.weight.sum() * 0.0
        hidden = self.encode(tokens[:, :-1], time_days[:, :-1])
        targets = tokens[:, 1:]
        logits = self.language_head(hidden)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=PAD_ID,
            reduction="sum",
        )
        return loss / targets.ne(PAD_ID).sum().clamp_min(1)


class BagOfEventsModel(nn.Module):
    """A deliberately small order-independent baseline with the same interface.

    Sum event embeddings, divide by sqrt(event count), and apply a linear head
    alongside static features.  PAD/BOS do not contribute.  Repeated events
    therefore retain frequency information without unbounded sequence scale.
    No future timestamps or external weights are used.
    """

    def __init__(
        self,
        vocab_size: int,
        n_targets: int,
        n_static: int,
        max_events: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if vocab_size < 4 or n_targets < 1 or n_static < 0 or max_events < 1:
            raise ValueError("Invalid vocabulary, target, static, or sequence size")
        self.max_events = max_events
        self.n_static = n_static
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(d_model + n_static, n_targets)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        _validate_shapes(tokens, time_days, static, self.max_events, self.n_static)
        event_mask = tokens.ne(PAD_ID) & tokens.ne(BOS_ID)
        vectors = self.token_embedding(tokens) * event_mask.unsqueeze(-1)
        scale = event_mask.sum(dim=1, keepdim=True).clamp_min(1).sqrt()
        patient_state = vectors.sum(dim=1) / scale
        combined = torch.cat((patient_state, static.to(patient_state.dtype)), dim=-1)
        return self.classification_head(self.dropout(combined))

    def next_event_loss(
        self, tokens: Tensor, time_days: Tensor, static: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError("Next-event pretraining requires the Transformer")


def build_model(model_config: Mapping[str, Any]) -> nn.Module:
    """Reconstruct a model from the JSON-compatible config in its checkpoint."""
    config = dict(model_config)
    model_type = config.pop("model_type", "transformer")
    prediction_offset = config.pop("prediction_logit_offset", None)
    constructors = {"transformer": TimelineTransformer, "bag": BagOfEventsModel}
    if model_type in {"pooled_transformer", "summary_mlp", "summary_linear"}:
        from .enhanced_models import PooledTimelineTransformer, SummaryMLP, SummaryLinear
        constructors.update(pooled_transformer=PooledTimelineTransformer,
                            summary_mlp=SummaryMLP, summary_linear=SummaryLinear)
    if model_type in {"compressed_transformer", "hierarchical_transformer"}:
        from .visit_models import CompressedTimelineTransformer, HierarchicalTimelineTransformer
        constructors.update(compressed_transformer=CompressedTimelineTransformer,
                            hierarchical_transformer=HierarchicalTimelineTransformer)
    if model_type == "experimental_bridge":
        from .experimental_bridge import ControlledTimelineTransformer
        constructors[model_type] = ControlledTimelineTransformer
    if model_type not in constructors:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    model = constructors[model_type](**config)
    if prediction_offset is not None:
        offset = torch.as_tensor(prediction_offset, dtype=torch.float32)
        if offset.shape != (config["n_targets"],) or not torch.isfinite(offset).all():
            raise ValueError("Prediction logit offset must have one finite value per target")
        model.register_buffer("prediction_logit_offset", offset)
    return model
