"""CPU checks for fair date compression and safe local/global aggregation."""

import pytest
import torch
from torch.nn import functional as F

from ehr_forecast.visit_models import CompressedTimelineTransformer, HierarchicalTimelineTransformer


CONSTRUCTORS = [CompressedTimelineTransformer, HierarchicalTimelineTransformer]


@pytest.fixture
def inputs():
    torch.set_num_threads(1)
    tokens = torch.tensor([[[4, 5, 6, 0], [7, 8, 0, 0], [9, 0, 0, 0], [10, 11, 0, 0], [0, 0, 0, 0]],
                           [[12, 13, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    times = torch.tensor([[1000., 200., 40., 10., 0.], [30., 0., 0., 0., 0.]])
    static = torch.tensor([[.3, 1.], [-.7, 0.]])
    return tokens, times, static


def make(constructor, **updates):
    config = dict(vocab_size=24, n_targets=3, n_static=2, max_events=12,
                  d_model=16, n_heads=4, n_layers=3, dropout=0., window_size=3,
                  max_unit_events=8)
    config.update(updates)
    return constructor(**config)


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_finite_logits_and_gradients(constructor, inputs):
    model = make(constructor)
    tokens, times, static = inputs
    logits = model(tokens, times, static)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert torch.isfinite(model.classification_head.weight.grad).all()
    assert model.token_embedding.weight.grad.abs().sum() > 0
    if isinstance(model, HierarchicalTimelineTransformer):
        assert model.local_encoder.layers[0].linear1.weight.grad.abs().sum() > 0
        assert model.global_encoder.layers[0].linear1.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_event_order_in_one_date_is_irrelevant(constructor, inputs):
    model = make(constructor).eval()
    tokens, times, static = inputs
    shuffled = tokens[..., torch.tensor([2, 0, 3, 1])]
    with torch.no_grad():
        torch.testing.assert_close(model(tokens, times, static), model(shuffled, times, static), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_unit_and_event_padding_are_irrelevant(constructor, inputs):
    model = make(constructor).eval()
    tokens, times, static = inputs
    padded = F.pad(tokens, (0, 3, 0, 6))
    padded_times = F.pad(times, (0, 6), value=99999.)
    with torch.no_grad():
        torch.testing.assert_close(model(tokens, times, static), model(padded, padded_times, static), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_empty_histories_and_empty_local_windows_are_safe(constructor):
    model = make(constructor).eval()
    tokens = torch.zeros(2, 12, 4, dtype=torch.long)
    times = torch.full((2, 12), 9999.)
    static = torch.zeros(2, 2)
    # Both the supported BOS sentinel and defensive all-PAD fallback are finite.
    tokens[0, 0, 0] = 2
    with torch.no_grad():
        before = model(tokens, times, static)
        assert torch.isfinite(before).all()
        times[1] = 0
        torch.testing.assert_close(before[1], model(tokens, times, static)[1])


def test_flat_and_hierarchical_budgets_and_common_initialization_match():
    torch.manual_seed(123)
    flat = make(CompressedTimelineTransformer)
    flat_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    hierarchy = make(HierarchicalTimelineTransformer)
    torch.testing.assert_close(torch.get_rng_state(), flat_rng, atol=0, rtol=0)
    assert sum(p.numel() for p in flat.parameters()) == sum(p.numel() for p in hierarchy.parameters())
    for name in ("token_embedding.weight", "position_embedding.weight", "time_projection.weight",
                 "count_projection.weight", "classification_head.weight", "readout.0.weight"):
        torch.testing.assert_close(flat.state_dict()[name], hierarchy.state_dict()[name], atol=0, rtol=0)
    hierarchical_layers = [*hierarchy.local_encoder.layers, *hierarchy.global_encoder.layers]
    for flat_layer, hierarchical_layer in zip(flat.backbone.layers, hierarchical_layers):
        for name, value in flat_layer.state_dict().items():
            torch.testing.assert_close(value, hierarchical_layer.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_raw_one_event_per_unit_and_objective_contract(constructor, inputs):
    model = make(constructor)
    tokens, times, static = inputs
    assert torch.isfinite(model(tokens[..., :1], times, static)).all()
    with pytest.raises(NotImplementedError):
        model.next_event_loss(tokens, times, static)
    with pytest.raises(TypeError):
        model(tokens.float(), times, static)
    with pytest.raises(ValueError):
        model(tokens[:, 0], times, static)
    with pytest.raises(ValueError):
        model(tokens, times[:, :2], static)
    with pytest.raises(ValueError):
        model(tokens, times, static[:, :1])


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_date_order_and_static_features_can_change_output(constructor, inputs):
    model = make(constructor).eval()
    tokens, times, static = inputs
    permutation = torch.tensor([3, 2, 1, 0, 4])
    with torch.no_grad():
        original = model(tokens, times, static)
        changed_order = model(tokens[:, permutation], times[:, permutation], static)
        changed_static = model(tokens, times, static + 1)
    assert not torch.allclose(original[0], changed_order[0])
    assert not torch.allclose(original, changed_static)


def test_hierarchy_rejects_unmatched_layer_budget():
    with pytest.raises(ValueError, match="sum to n_layers"):
        make(HierarchicalTimelineTransformer, local_layers=2, global_layers=2)


def _legacy_nonoverlap_logits(model, tokens, times, static):
    """The former reshape path is an independent default-behavior reference."""
    from ehr_forecast.visit_models import _masked_mean
    hidden, mask = model.embed_dates(tokens, times)
    batch, units, width = hidden.shape
    windows = (units + model.window_size - 1) // model.window_size
    padded = windows * model.window_size
    hidden = F.pad(hidden, (0, 0, 0, padded - units))
    mask = F.pad(mask, (0, padded - units), value=False)
    blocks = hidden.reshape(batch * windows, model.window_size, width)
    block_mask = mask.reshape(batch * windows, model.window_size)
    active = block_mask.any(dim=1)
    encoded = model.local_encoder(blocks[active], src_key_padding_mask=~block_mask[active])
    pooled = hidden.new_zeros(batch * windows, width).index_copy(
        0, active.nonzero(as_tuple=True)[0], _masked_mean(encoded, block_mask[active])
    ).reshape(batch, windows, width)
    global_mask = active.reshape(batch, windows)
    hidden = model.global_encoder(pooled, src_key_padding_mask=~global_mask)
    return model._classify(hidden, global_mask, static)


def test_default_stride_preserves_legacy_nonoverlap_path(inputs):
    model = make(HierarchicalTimelineTransformer).eval()
    tokens, times, static = inputs
    with torch.no_grad():
        torch.testing.assert_close(model(tokens, times, static),
                                   _legacy_nonoverlap_logits(model, tokens, times, static), atol=0, rtol=0)


def test_stride_does_not_change_parameter_initialization():
    torch.manual_seed(81)
    nonoverlap = make(HierarchicalTimelineTransformer)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(81)
    overlap = make(HierarchicalTimelineTransformer, window_stride=1)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert nonoverlap.state_dict().keys() == overlap.state_dict().keys()
    for name, weight in nonoverlap.state_dict().items():
        torch.testing.assert_close(weight, overlap.state_dict()[name], atol=0, rtol=0)
    torch.manual_seed(81)
    explicit = make(HierarchicalTimelineTransformer, window_stride=3)
    for name, weight in nonoverlap.state_dict().items():
        torch.testing.assert_close(weight, explicit.state_dict()[name], atol=0, rtol=0)


def test_overlap_padding_and_gradient_safety(inputs):
    model = make(HierarchicalTimelineTransformer, window_stride=1).eval()
    tokens, times, static = inputs
    with torch.no_grad():
        original = model(tokens, times, static)
        padded = model(F.pad(tokens, (0, 2, 0, 6)), F.pad(times, (0, 6), value=99999), static)
        torch.testing.assert_close(original, padded, atol=1e-6, rtol=1e-6)
    loss = model(tokens, times, static).square().mean()
    loss.backward()
    for name in ("local_encoder", "global_encoder"):
        gradient = getattr(model, name).layers[0].linear1.weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("stride", [0, -1, 4])
def test_invalid_window_stride_is_rejected(stride):
    with pytest.raises(ValueError, match="between 1 and window_size"):
        make(HierarchicalTimelineTransformer, window_stride=stride)


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_bag_normalization_preserves_initial_weights_and_raw_control(constructor, inputs):
    torch.manual_seed(703)
    default = make(constructor).eval()
    default_rng = torch.get_rng_state().clone()
    torch.manual_seed(703)
    explicit_mean = make(constructor, bag_normalization="mean").eval()
    torch.testing.assert_close(torch.get_rng_state(), default_rng, atol=0, rtol=0)
    torch.manual_seed(703)
    sqrt = make(constructor, bag_normalization="sqrt").eval()
    torch.testing.assert_close(torch.get_rng_state(), default_rng, atol=0, rtol=0)
    assert sum(p.numel() for p in default.parameters()) == sum(p.numel() for p in sqrt.parameters())
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit_mean.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(value, sqrt.state_dict()[name], atol=0, rtol=0)
    tokens, times, static = inputs
    with torch.no_grad():
        torch.testing.assert_close(default(tokens, times, static), explicit_mean(tokens, times, static), atol=0, rtol=0)
        # E=1 contains exactly one event per real unit, so sum/sqrt(1) and mean
        # are bitwise identical throughout the raw-event control's forward.
        torch.testing.assert_close(default(tokens[..., :1], times, static),
                                   sqrt(tokens[..., :1], times, static), atol=0, rtol=0)


def test_sqrt_bag_normalization_math_and_zero_padding():
    model = make(CompressedTimelineTransformer, bag_normalization="sqrt").eval()
    with torch.no_grad():
        model.position_embedding.weight.zero_()
        for layer in (model.time_projection, model.count_projection):
            layer.weight.zero_()
            layer.bias.zero_()
        vector = torch.arange(1, model.d_model + 1, dtype=torch.float32)
        model.token_embedding.weight[4].copy_(vector)
    tokens = torch.tensor([[[4, 4, 0, 0], [4, 4, 4, 4], [0, 0, 0, 0]]])
    hidden, mask = model.embed_dates(tokens, torch.zeros(1, 3))
    torch.testing.assert_close(hidden[0, 0], vector * 2**.5)
    torch.testing.assert_close(hidden[0, 1], vector * 2)
    torch.testing.assert_close(hidden[0, 2], torch.zeros_like(vector))
    assert mask.tolist() == [[True, True, False]]


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_sqrt_pooling_permutation_padding_empty_histories_and_gradients(constructor, inputs):
    model = make(constructor, bag_normalization="sqrt", window_stride=1).eval()
    tokens, times, static = inputs
    with torch.no_grad():
        original = model(tokens, times, static)
        permuted = model(tokens[..., torch.tensor([2, 0, 3, 1])], times, static)
        padded = model(F.pad(tokens, (0, 2, 0, 6)), F.pad(times, (0, 6), value=99999), static)
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original, padded, atol=1e-6, rtol=1e-6)
        assert torch.isfinite(model(torch.zeros_like(tokens), times, static)).all()
    model(tokens, times, static).square().mean().backward()
    gradient = model.token_embedding.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_invalid_bag_normalization_is_rejected():
    for constructor in CONSTRUCTORS:
        with pytest.raises(ValueError, match="bag_normalization"):
            make(constructor, bag_normalization="sum")
