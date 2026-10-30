"""Behavioral model tests; all tests are intentionally tiny and CPU-only."""

from __future__ import annotations

import unittest

import torch

from ehr_forecast.models import BagOfEventsModel, TimelineTransformer, build_model


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        torch.set_num_threads(1)
        self.config = dict(
            vocab_size=20,
            n_targets=4,
            n_static=2,
            max_events=8,
            d_model=16,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
        self.tokens = torch.tensor([[2, 5, 6, 7, 0], [2, 9, 0, 0, 0]])
        self.times = torch.tensor([[0.0, 70, 30, 10, 0], [0.0, 50, 0, 0, 0]])
        self.static = torch.tensor([[0.2, 1.0], [-0.3, 0.0]])

    def test_output_shape_and_finite_gradients(self) -> None:
        for constructor in (TimelineTransformer, BagOfEventsModel):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config)
                logits = model(self.tokens, self.times, self.static)
                self.assertEqual(logits.shape, (2, 4))
                self.assertTrue(torch.isfinite(logits).all())
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits)
                ).backward()
                self.assertTrue(torch.isfinite(model.classification_head.weight.grad).all())
                self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0)

    def test_suffix_cannot_change_earlier_hidden_states(self) -> None:
        model = TimelineTransformer(**self.config).eval()
        altered_tokens = self.tokens.clone()
        altered_times = self.times.clone()
        altered_tokens[0, 3] = 15
        altered_times[0, 3] = 9999
        with torch.no_grad():
            original = model.encode(self.tokens, self.times)
            altered = model.encode(altered_tokens, altered_times)
        torch.testing.assert_close(original[:, :3], altered[:, :3], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(original[0, 3], altered[0, 3]))

    def test_extra_right_padding_cannot_change_predictions(self) -> None:
        for constructor in (TimelineTransformer, BagOfEventsModel):
            with self.subTest(model=constructor.__name__):
                model = constructor(**self.config).eval()
                padded_tokens = torch.nn.functional.pad(self.tokens, (0, 3))
                # Even nonzero padding timestamps must have no influence.
                padded_times = torch.nn.functional.pad(self.times, (0, 3), value=9000)
                with torch.no_grad():
                    before = model(self.tokens, self.times, self.static)
                    after = model(padded_tokens, padded_times, self.static)
                torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)

    def test_next_event_loss_padding_and_gradients(self) -> None:
        model = TimelineTransformer(**self.config).eval()
        loss = model.next_event_loss(self.tokens, self.times, self.static)
        padded_loss = model.next_event_loss(
            torch.nn.functional.pad(self.tokens, (0, 3)),
            torch.nn.functional.pad(self.times, (0, 3)),
            self.static,
        )
        torch.testing.assert_close(loss, padded_loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.language_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0)

    def test_bos_only_batch_has_finite_differentiable_zero_loss(self) -> None:
        model = TimelineTransformer(**self.config)
        for tokens in (torch.tensor([[2], [2]]), torch.tensor([[2, 0], [2, 0]])):
            loss = model.next_event_loss(tokens, torch.zeros_like(tokens).float(), self.static)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(model(tokens, torch.zeros_like(tokens).float(), self.static)).all())

    def test_bag_model_is_order_invariant(self) -> None:
        model = BagOfEventsModel(**self.config).eval()
        reordered = self.tokens.clone()
        reordered[0, 1:4] = self.tokens[0, torch.tensor([3, 1, 2])]
        with torch.no_grad():
            torch.testing.assert_close(
                model(self.tokens, self.times, self.static),
                model(reordered, self.times, self.static),
                atol=1e-6,
                rtol=1e-6,
            )

    def test_build_and_invalid_input(self) -> None:
        self.assertIsInstance(build_model({"model_type": "bag", **self.config}), BagOfEventsModel)
        self.assertIsInstance(build_model(self.config), TimelineTransformer)
        with self.assertRaises(ValueError):
            build_model({"model_type": "unknown", **self.config})
        model = TimelineTransformer(**self.config)
        with self.assertRaises(ValueError):
            model(self.tokens, self.times[:, :2], self.static)
        with self.assertRaises(ValueError):
            model(self.tokens, self.times, self.static[:, :1])


if __name__ == "__main__":
    unittest.main()
