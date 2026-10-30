"""Controlled changes around the unchanged legacy patient-timeline model.

The legacy constructor is always run in full, including its unused language
head.  A new readout is constructed in a CPU RNG fork and initialized from a
local generator, so it cannot change common backbone weights or the CPU/CUDA
RNG state that subsequent training observes.  This controls initialization;
different attention/readout graphs can still consume different dropout draws.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .models import BOS_ID, PAD_ID, TimelineTransformer, _validate_shapes


class ControlledTimelineTransformer(TimelineTransformer):
    """A two-factor bridge: attention direction and patient readout only.

    Input events, BOS, positions, time features, output normalization and the
    legacy language head are unchanged. A [B,L,1] tensor is only squeezed into
    the exact legacy [B,L] representation; it is never grouped or reordered.
    """

    READOUT_SEED_OFFSET = 17_041

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
        attention_mode: str = "causal",
        readout_mode: str = "legacy",
    ) -> None:
        if attention_mode not in {"causal", "bidirectional"}:
            raise ValueError("attention_mode must be 'causal' or 'bidirectional'")
        if readout_mode not in {"legacy", "pooled_mlp"}:
            raise ValueError("readout_mode must be 'legacy' or 'pooled_mlp'")

        # Do not streamline this constructor: its complete initialization draw
        # sequence is the experimental anchor, even for unused legacy modules.
        super().__init__(
            vocab_size=vocab_size, n_targets=n_targets, n_static=n_static,
            max_events=max_events, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )
        self.attention_mode = attention_mode
        self.readout_mode = readout_mode

        if readout_mode == "pooled_mlp":
            # Explicit CPU construction plus devices=[] avoids touching CUDA
            # RNG. Do not use torch.manual_seed here: it also seeds CUDA.
            head_seed = (torch.initial_seed() + self.READOUT_SEED_OFFSET) % (2**63 - 1)
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator(device="cpu").manual_seed(head_seed)
                self.readout = nn.Sequential(
                    nn.Linear(2 * d_model + n_static, d_model, device="cpu"),
                    nn.LayerNorm(d_model, device="cpu"),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.classification_head = nn.Linear(d_model, n_targets, device="cpu")
                for module in (self.readout[0], self.classification_head):
                    nn.init.xavier_uniform_(module.weight, generator=generator)
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _legacy_input(tokens: Tensor, time_days: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim == 3:
            if tokens.shape[-1] != 1:
                raise ValueError("The bridge adapter requires exactly one event per unit (E=1)")
            tokens = tokens.squeeze(-1)
        return tokens, time_days

    def encode(self, tokens: Tensor, time_days: Tensor) -> Tensor:
        tokens, time_days = self._legacy_input(tokens, time_days)
        if self.attention_mode == "causal":
            return super().encode(tokens, time_days)

        # Exactly the legacy embedding/time/position computation. Only the
        # causal attention mask is removed; supplied history is pre-anchor.
        _validate_shapes(tokens, time_days, None, self.max_events, self.n_static)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        days = time_days.to(dtype=self.token_embedding.weight.dtype).clamp_min(0)
        time_features = torch.stack(
            (torch.log1p(days) / 10.0,
             (days / 365.25).clamp_max(120) / 120.0),
            dim=-1,
        )
        hidden = (
            self.token_embedding(tokens)
            + self.position_embedding(positions).unsqueeze(0)
            + self.time_projection(time_features)
        )
        hidden = self.backbone(
            self.input_dropout(hidden),
            src_key_padding_mask=tokens.eq(PAD_ID),
            is_causal=False,
        )
        return self.output_norm(hidden)

    def forward(self, tokens: Tensor, time_days: Tensor, static: Tensor) -> Tensor:
        tokens, time_days = self._legacy_input(tokens, time_days)
        if self.readout_mode == "legacy":
            return super().forward(tokens, time_days, static)

        _validate_shapes(tokens, time_days, static, self.max_events, self.n_static)
        hidden = self.encode(tokens, time_days)
        clinical_mask = tokens.ne(PAD_ID) & tokens.ne(BOS_ID)
        clinical_mean = (hidden * clinical_mask.unsqueeze(-1)).sum(dim=1)
        clinical_mean = clinical_mean / clinical_mask.sum(dim=1, keepdim=True).clamp_min(1)
        # Retain the legacy last-position convention; preparation guarantees
        # right padding and BOS even for patients with no clinical events.
        last_positions = tokens.ne(PAD_ID).sum(dim=1).sub(1).clamp_min(0)
        last_state = hidden[
            torch.arange(tokens.shape[0], device=tokens.device), last_positions
        ]
        combined = torch.cat((clinical_mean, last_state, static.to(hidden.dtype)), dim=-1)
        return self.classification_head(self.readout(combined))

    def next_event_loss(
        self, tokens: Tensor, time_days: Tensor, static: Tensor | None = None
    ) -> Tensor:
        if self.attention_mode == "bidirectional":
            raise NotImplementedError(
                "Bidirectional bridge attention is unsafe for next-event pretraining: "
                "it could expose the target token. Use supervised classification only."
            )
        tokens, time_days = self._legacy_input(tokens, time_days)
        return super().next_event_loss(tokens, time_days, static)
