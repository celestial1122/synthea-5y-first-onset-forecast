import itertools
import json

import pytest
import torch

from ranking_study.objectives import (
    plackett_luce_log_prob, rank_loss as original_rank_loss,
)
from ranking_study.scale_invariant import normalize_scores, rank_loss


METHODS = ("smoothap_si", "rloo_ap_si", "grpo_ap_si")


@pytest.fixture(scope="module", autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example():
    logits = torch.tensor([[.4, -.8], [.9, .2], [-.2, .7], [.1, -.3], [.8, .1]], dtype=torch.float64)
    labels = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    eligible = torch.tensor([[1, 1], [1, 0], [1, 1], [0, 1], [1, 1]], dtype=torch.bool)
    reference = torch.stack([logits[:, c][eligible[:, c]].std(correction=0) for c in range(2)])
    return logits, labels, eligible, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_theta0_forward_is_bitwise_identical(dtype):
    logits, _, eligible, _ = example()
    logits = logits.to(dtype)
    reference = torch.stack([logits[:, c][eligible[:, c]].std(correction=0) for c in range(2)])
    actual = normalize_scores(logits, eligible, reference)
    assert torch.equal(actual, logits)


def test_positive_affine_transform_changes_normalized_scores_only_by_common_offsets():
    logits, _, eligible, reference = example()
    scales = torch.tensor([2.7, .35], dtype=torch.float64)
    offsets = torch.tensor([-1.4, 3.2], dtype=torch.float64)
    transformed = logits * scales + offsets
    first = normalize_scores(logits, eligible, reference)
    second = normalize_scores(transformed, eligible, reference)
    means = torch.stack([logits[:, c][eligible[:, c]].mean() for c in range(2)])
    expected_offsets = (scales - 1) * means + offsets
    torch.testing.assert_close(second - first, expected_offsets.expand_as(first), atol=1e-14, rtol=1e-14)
    # Exhaust the full PL distribution, rather than checking only greedy ranks.
    for c in range(2):
        first_scores, second_scores = first[:, c][eligible[:, c]], second[:, c][eligible[:, c]]
        orders = torch.tensor(list(itertools.permutations(range(first_scores.numel()))))
        torch.testing.assert_close(plackett_luce_log_prob(first_scores / .6, orders),
                                   plackett_luce_log_prob(second_scores / .6, orders), atol=1e-13, rtol=1e-13)


def test_normalizer_current_std_derivative_is_not_detached():
    logits = torch.tensor([[.2], [.9], [-.5], [.1]], dtype=torch.float64, requires_grad=True)
    eligible = torch.ones_like(logits, dtype=torch.bool)
    reference = logits.detach().std(dim=0, correction=0)
    weights = torch.tensor([[.3], [1.2], [-.7], [.4]], dtype=torch.float64)
    normalized = normalize_scores(logits, eligible, reference)
    grad = torch.autograd.grad((normalized * weights).sum(), logits)[0]
    centered = logits.detach() - logits.detach().mean()
    expected = weights - centered * (weights * centered).sum() / centered.square().sum()
    torch.testing.assert_close(grad, expected, atol=1e-13, rtol=1e-13)
    assert not torch.allclose(grad, weights)


@pytest.mark.parametrize("method", METHODS)
def test_theta0_loss_policy_rewards_and_stream_match_old_objective(method):
    logits, labels, eligible, reference = example()
    base = {"smoothap_si": "smoothap", "rloo_ap_si": "rloo_ap", "grpo_ap_si": "grpo_ap"}[method]
    first_generator = torch.Generator().manual_seed(993)
    second_generator = torch.Generator().manual_seed(993)
    actual, actual_stats = rank_loss(logits, labels, eligible, method, .7,
        generator=first_generator, reference_std=reference)
    expected, expected_stats = original_rank_loss(logits, labels, eligible, base, .7,
        generator=second_generator)
    assert torch.equal(actual, expected)
    assert torch.equal(first_generator.get_state(), second_generator.get_state())
    assert actual_stats["reward_mean"] == expected_stats["reward_mean"]
    assert actual_stats["reward_std"] == expected_stats["reward_std"]
    assert actual_stats["scale_ratio_min"] == actual_stats["scale_ratio_max"] == 1.
    assert actual_stats["score_normalization_full_derivatives"]


@pytest.mark.parametrize("method", METHODS)
def test_rank_objective_and_projected_gradient_are_affine_scale_invariant(method):
    raw, labels, eligible, reference = example()
    scales = torch.tensor([2.7, .35], dtype=torch.float64)
    offsets = torch.tensor([-1.4, 3.2], dtype=torch.float64)
    outputs = []
    for values in (raw.clone(), raw * scales + offsets):
        values.requires_grad_()
        loss, stats = rank_loss(values, labels, eligible, method, torch.tensor([.6, .8]),
            generator=torch.Generator().manual_seed(63), reference_std=reference)
        gradient = torch.autograd.grad(loss, values)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert torch.all(gradient[~eligible] == 0)
        torch.testing.assert_close(gradient.sum(dim=0), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
        # Euler's degree-zero identity: no first-order ranking signal can be
        # obtained merely by scaling every raw patient score within a disease.
        torch.testing.assert_close((gradient * values.detach()).sum(dim=0),
                                   torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
        json.dumps(stats, allow_nan=False)
        outputs.append((loss.detach(), gradient))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(outputs[0][1], outputs[1][1] * scales, atol=1e-12, rtol=1e-12)


def test_reference_std_is_fixed_and_prior_logits_do_not_enter_moments():
    raw, labels, eligible, reference = example()
    raw.requires_grad_()
    reference.requires_grad_()
    loss, _ = rank_loss(raw, labels, eligible, "smoothap_si", .6, reference_std=reference)
    gradient, ref_gradient = torch.autograd.grad(loss, (raw, reference), allow_unused=True)
    assert ref_gradient is None
    assert torch.all(gradient[~eligible] == 0)
    changed = raw.detach().clone()
    changed[~eligible] += 10000.
    z1 = normalize_scores(raw, eligible, reference)
    z2 = normalize_scores(changed, eligible, reference)
    assert torch.equal(z1[eligible], z2[eligible])


@pytest.mark.parametrize("chunk_size", [1, 2, 128])
def test_smooth_ap_chunked_full_gradients_are_finite_and_identical(chunk_size):
    raw, labels, eligible, reference = example()
    values = raw.clone().requires_grad_()
    other = raw.clone().requires_grad_()
    first, _ = rank_loss(values, labels, eligible, "smoothap_si", .6,
                         reference_std=reference, chunk_size=chunk_size)
    second, _ = rank_loss(other, labels, eligible, "smoothap_si", .6,
                          reference_std=reference, chunk_size=128)
    first_grad = torch.autograd.grad(first, values)[0]
    second_grad = torch.autograd.grad(second, other)[0]
    torch.testing.assert_close(first, second, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(first_grad, second_grad, atol=1e-13, rtol=1e-13)


def test_normalizer_and_smooth_ap_pass_double_gradcheck():
    raw, labels, eligible, reference = example()
    raw.requires_grad_()
    assert torch.autograd.gradcheck(lambda values: normalize_scores(values, eligible, reference),
                                   (raw,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradcheck(lambda values: rank_loss(values, labels, eligible, "smoothap_si", .6,
                                                           reference_std=reference)[0],
                                   (raw,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_degenerate_scales_raise_instead_of_silently_changing_objective():
    raw, labels, eligible, reference = example()
    for value in (0., 1e-6, -1., float("nan")):
        invalid = reference.clone()
        invalid[0] = value
        with pytest.raises(ValueError):
            normalize_scores(raw, eligible, invalid)
    constant = raw.clone()
    constant[:, 0] = .5
    with pytest.raises(ValueError, match="current"):
        normalize_scores(constant, eligible, reference)
    too_small = raw.clone()
    too_small[:, 0] *= 1e-7
    with pytest.raises(ValueError, match="current"):
        normalize_scores(too_small, eligible, reference)
    too_few = eligible.clone()
    too_few[:, 0] = False
    too_few[0, 0] = True
    with pytest.raises(ValueError, match="at least two"):
        normalize_scores(raw, too_few, reference)
    for invalid_eps in (0., -1., float("nan")):
        with pytest.raises(ValueError):
            normalize_scores(raw, eligible, reference, eps=invalid_eps)


def test_bce_control_is_raw_zero_rank_component_and_has_no_normalization():
    raw, labels, eligible, reference = example()
    raw.requires_grad_()
    loss, stats = rank_loss(raw, labels, eligible, "bce", reference_std=reference)
    assert loss.item() == 0
    assert stats["score_normalization"] is None
    assert torch.all(torch.autograd.grad(loss, raw)[0] == 0)
    with pytest.raises(ValueError, match="method"):
        rank_loss(raw, labels, eligible, "smoothap", reference_std=reference)
