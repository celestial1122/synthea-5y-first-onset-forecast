"""CPU-only behavioral tests for summary models and pooled classification."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from ehr_forecast.enhanced_models import (
    PooledTimelineTransformer,
    SummaryLinear,
    SummaryMLP,
)
from ehr_forecast.models import TimelineTransformer


class EnhancedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        torch.set_num_threads(1)
        self.config = dict(
            vocab_size=24, n_targets=3, n_static=6, max_events=8,
            d_model=16, n_heads=4, n_layers=2, dropout=0.0,
        )
        self.tokens = torch.tensor([[2, 5, 7, 8, 0], [2, 9, 0, 0, 0]])
        self.times = torch.tensor([[0.0, 90, 40, 10, 0], [0.0, 30, 0, 0, 0]])
        self.static = torch.tensor([[0.2, 1.0, 0.0, 0.3, -0.5, 0.1], [-0.3, 0.0, 1.0, 0.5, 0.2, -0.7]])

    def test_finite_predictions_and_gradients(self) -> None:
        for constructor in (SummaryMLP, SummaryLinear, PooledTimelineTransformer):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config)
                static = self.static.clone().requires_grad_()
                logits = model(self.tokens, self.times, static)
                self.assertEqual(logits.shape, (2, 3))
                self.assertTrue(torch.isfinite(logits).all())
                F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
                self.assertIsInstance(model.classification_head, nn.Linear)
                self.assertTrue(torch.isfinite(model.classification_head.weight.grad).all())
                self.assertGreater(static.grad.abs().sum().item(), 0)
                if isinstance(model, PooledTimelineTransformer):
                    self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0)

    def test_summary_predictions_ignore_timeline_content(self) -> None:
        for constructor in (SummaryMLP, SummaryLinear):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config).eval()
                alternative_tokens = self.tokens.clone()
                alternative_tokens[:, 1:] = 15
                alternative_times = torch.full_like(self.times, 9999.0)
                with torch.no_grad():
                    torch.testing.assert_close(
                        model(self.tokens, self.times, self.static),
                        model(alternative_tokens, alternative_times, self.static),
                        atol=0, rtol=0,
                    )

    def test_summary_features_change_predictions(self) -> None:
        for constructor in (SummaryMLP, SummaryLinear, PooledTimelineTransformer):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config).eval()
                changed = self.static.clone()
                changed[:, :3] += torch.tensor([2.0, -1.0, 0.7])
                with torch.no_grad():
                    before = model(self.tokens, self.times, self.static)
                    after = model(self.tokens, self.times, changed)
                self.assertFalse(torch.allclose(before, after))

    def test_padding_does_not_change_predictions(self) -> None:
        for constructor in (SummaryMLP, SummaryLinear, PooledTimelineTransformer):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config).eval()
                with torch.no_grad():
                    before = model(self.tokens, self.times, self.static)
                    after = model(F.pad(self.tokens, (0, 3)), F.pad(self.times, (0, 3), value=9999), self.static)
                torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)

    def test_bos_only_empty_history_is_finite(self) -> None:
        for constructor in (SummaryMLP, SummaryLinear, PooledTimelineTransformer):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config).eval()
                for tokens in (torch.tensor([[2], [2]]), torch.tensor([[2, 0], [2, 0]])):
                    times = torch.zeros_like(tokens).float()
                    self.assertTrue(torch.isfinite(model(tokens, times, self.static)).all())
                    if isinstance(model, PooledTimelineTransformer):
                        self.assertEqual(model.next_event_loss(tokens, times, self.static).item(), 0)

    def test_pooled_readout_preserves_backbone_and_pretraining(self) -> None:
        torch.manual_seed(91)
        baseline = TimelineTransformer(**self.config).eval()
        torch.manual_seed(91)
        pooled = PooledTimelineTransformer(**self.config).eval()
        with torch.no_grad():
            torch.testing.assert_close(baseline.encode(self.tokens, self.times), pooled.encode(self.tokens, self.times), atol=0, rtol=0)
            torch.testing.assert_close(
                baseline.next_event_loss(self.tokens, self.times, self.static),
                pooled.next_event_loss(self.tokens, self.times, self.static), atol=0, rtol=0,
            )

    def test_pooled_backbone_remains_causal(self) -> None:
        model = PooledTimelineTransformer(**self.config).eval()
        altered = self.tokens.clone()
        altered[0, 3] = 21
        with torch.no_grad():
            before = model.encode(self.tokens, self.times)
            after = model.encode(altered, self.times)
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(before[0, 3], after[0, 3]))

    def test_summary_shape_checks_and_objective(self) -> None:
        for constructor in (SummaryMLP, SummaryLinear):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config)
                with self.assertRaises(ValueError):
                    model(self.tokens, self.times, self.static[:, :2])
                with self.assertRaises(ValueError):
                    model(self.tokens, self.times[:, :2], self.static)
                with self.assertRaises(TypeError):
                    model(self.tokens.float(), self.times, self.static)
                with self.assertRaises(ValueError):
                    model(self.tokens, self.times, None)
                with self.assertRaises(NotImplementedError):
                    model.next_event_loss(self.tokens, self.times, self.static)
                with self.assertRaises(ValueError):
                    constructor(**{**self.config, "n_static": 0})


if __name__ == "__main__":
    unittest.main()
