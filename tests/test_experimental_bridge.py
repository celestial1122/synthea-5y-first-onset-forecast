"""CPU-only experimental gates, not predictive-performance assertions."""

from itertools import product
from unittest.mock import patch

import pytest
import torch
from torch import nn

from ehr_forecast.experimental_bridge import ControlledTimelineTransformer
from ehr_forecast.models import TimelineTransformer


CONFIG = dict(vocab_size=23, n_targets=3, n_static=2, max_events=8,
              d_model=16, n_heads=4, n_layers=2, dropout=0.2)


def _cpu_seed(seed):
    # Tests must not initialize a GPU or change its seed as a side effect.
    torch.random.default_generator.manual_seed(seed)


def _models(seed=17, **updates):
    _cpu_seed(seed)
    original = TimelineTransformer(**{**CONFIG, **updates})
    original_rng = torch.get_rng_state().clone()
    _cpu_seed(seed)
    bridge = ControlledTimelineTransformer(**{**CONFIG, **updates})
    bridge_rng = torch.get_rng_state().clone()
    return original, bridge, original_rng, bridge_rng


def _batch():
    return (
        torch.tensor([[2, 4, 5, 0, 0], [2, 6, 7, 8, 9], [2, 0, 0, 0, 0]]),
        torch.tensor([[0., 90., 20., 0., 0.], [0., 200., 70., 10., 1.], [0., 0., 0., 0., 0.]]),
        torch.tensor([[.2, 1.], [-.3, 0.], [.7, 1.]]),
    )


def _exact(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [17, 42, 73])
def test_all_four_cells_keep_common_initial_state_and_rng(seed):
    _cpu_seed(seed)
    legacy = TimelineTransformer(**CONFIG)
    baseline_state = legacy.state_dict()
    baseline_rng = torch.get_rng_state().clone()
    pooled_states = []
    for attention, readout in product(("causal", "bidirectional"), ("legacy", "pooled_mlp")):
        _cpu_seed(seed)
        model = ControlledTimelineTransformer(**CONFIG, attention_mode=attention, readout_mode=readout)
        _exact(torch.get_rng_state(), baseline_rng)
        actual = model.state_dict()
        if readout == "legacy":
            assert actual.keys() == baseline_state.keys()
        for name, value in baseline_state.items():
            if readout == "pooled_mlp" and name.startswith("classification_head."):
                continue
            _exact(actual[name], value)
        if readout == "pooled_mlp":
            assert [type(layer).__name__ for layer in model.readout] == ["Linear", "LayerNorm", "GELU", "Dropout"]
            pooled_states.append({name: value.clone() for name, value in actual.items()
                                  if name.startswith(("readout.", "classification_head."))})
    assert pooled_states[0].keys() == pooled_states[1].keys()
    for name in pooled_states[0]:
        _exact(pooled_states[0][name], pooled_states[1][name])


def test_pooled_constructor_does_not_seed_or_initialize_cuda():
    _cpu_seed(42)
    with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA seed touched")), \
         patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA initialized")):
        model = ControlledTimelineTransformer(**CONFIG, readout_mode="pooled_mlp")
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.parametrize("adapter", [False, True])
def test_legacy_forward_loss_gradients_and_adamw_step_are_exact(adapter):
    legacy, bridge, rng, rng_bridge = _models()
    _exact(rng, rng_bridge)
    tokens, times, static = _batch()
    labels = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    eligible = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    models = [legacy, bridge]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01) for model in models]
    logits, losses = [], []
    for index, (model, optimizer) in enumerate(zip(models, optimizers)):
        torch.set_rng_state(rng.clone())
        values = model(tokens.unsqueeze(-1) if adapter and index else tokens, times, static)
        loss = (nn.functional.binary_cross_entropy_with_logits(values, labels, reduction="none") * eligible).sum() / eligible.sum()
        loss.backward()
        logits.append(values.detach())
        losses.append(loss.detach())
    _exact(logits[0], logits[1])
    _exact(losses[0], losses[1])
    legacy_params, bridge_params = dict(legacy.named_parameters()), dict(bridge.named_parameters())
    assert legacy_params.keys() == bridge_params.keys()
    for name in legacy_params:
        left, right = legacy_params[name].grad, bridge_params[name].grad
        if left is None:
            assert right is None
        else:
            _exact(left, right)
    for model, optimizer in zip(models, optimizers):
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    for name, value in legacy.state_dict().items():
        _exact(value, bridge.state_dict()[name])
    states = [optimizer.state_dict() for optimizer in optimizers]
    assert states[0]["param_groups"] == states[1]["param_groups"]
    assert states[0]["state"].keys() == states[1]["state"].keys()
    for parameter in states[0]["state"]:
        for name in states[0]["state"][parameter]:
            _exact(states[0]["state"][parameter][name], states[1]["state"][parameter][name])


def test_legacy_next_event_loss_and_checkpoint_roundtrip_are_exact(tmp_path):
    legacy, bridge, rng, _ = _models()
    tokens, times, static = _batch()
    torch.set_rng_state(rng.clone())
    expected = legacy.next_event_loss(tokens, times, static)
    torch.set_rng_state(rng.clone())
    actual = bridge.next_event_loss(tokens.unsqueeze(-1), times, static)
    _exact(expected, actual)
    path = tmp_path / "bridge.pt"
    torch.save({"config": CONFIG, "state": bridge.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = ControlledTimelineTransformer(**saved["config"])
    restored.load_state_dict(saved["state"], strict=True)
    legacy.eval()
    restored.eval()
    _exact(legacy(tokens, times, static), restored(tokens.unsqueeze(-1), times, static))


def test_causal_prefix_isolated_and_bidirectional_prefix_can_see_suffix():
    options = {**CONFIG, "dropout": 0.}
    _cpu_seed(73)
    causal = ControlledTimelineTransformer(**options).eval()
    _cpu_seed(73)
    bidirectional = ControlledTimelineTransformer(**options, attention_mode="bidirectional").eval()
    tokens = torch.tensor([[2, 4, 5, 6]])
    changed = torch.tensor([[2, 4, 7, 8]])
    times = torch.tensor([[0., 30., 20., 10.]])
    before, after = causal.encode(tokens, times), causal.encode(changed, times)
    _exact(before[:, :2], after[:, :2])
    before, after = bidirectional.encode(tokens, times), bidirectional.encode(changed, times)
    assert (before[:, :2] - after[:, :2]).abs().max() > 1e-5


@pytest.mark.parametrize("readout", ["legacy", "pooled_mlp"])
def test_bidirectional_next_event_pretraining_is_explicitly_rejected(readout):
    model = ControlledTimelineTransformer(**CONFIG, attention_mode="bidirectional", readout_mode=readout)
    tokens, times, static = _batch()
    with pytest.raises(NotImplementedError, match="expose the target token"):
        model.next_event_loss(tokens, times, static)


def test_pooled_clinical_mean_excludes_bos_and_padding():
    model = ControlledTimelineTransformer(**{**CONFIG, "dropout": 0.}, readout_mode="pooled_mlp").eval()
    tokens, times, static = _batch()
    captured = []
    handle = model.readout.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
    model(tokens, times, static)
    handle.remove()
    hidden = model.encode(tokens, times)
    d_model = CONFIG["d_model"]
    _exact(captured[0][0, :d_model], hidden[0, 1:3].mean(dim=0))
    _exact(captured[0][2, :d_model], torch.zeros(d_model))
    _exact(captured[0][:, -CONFIG["n_static"]:], static)
    assert torch.isfinite(captured[0]).all()


@pytest.mark.parametrize("keyword,value", [("attention_mode", "unknown"), ("readout_mode", "unknown")])
def test_invalid_modes_rejected(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        ControlledTimelineTransformer(**CONFIG, **{keyword: value})


def test_adapter_rejects_actual_event_bags():
    model = ControlledTimelineTransformer(**CONFIG)
    with pytest.raises(ValueError, match="exactly one event per unit"):
        model(torch.ones(2, 4, 2, dtype=torch.long), torch.zeros(2, 4), torch.zeros(2, 2))
