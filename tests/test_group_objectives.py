import itertools
import json

import pytest
import torch
from torch.nn import functional as F

from ranking_study.objectives import (
    group_advantages, macro_bce_logits, permutation_ap_rewards,
    plackett_luce_log_prob, rank_loss, sample_plackett_luce,
)


METHODS = ("ranknet", "listnet", "smoothap", "rloo_ap", "grpo_ap")


@pytest.fixture(scope="module", autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("n", [3, 4])
def test_full_pl_permutation_probabilities_sum_to_one(n):
    scores = torch.tensor([.6, -.8, .1, 1.4][:n], dtype=torch.float64, requires_grad=True)
    orders = torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long)
    log_prob = plackett_luce_log_prob(scores, orders)
    assert torch.exp(log_prob).sum().item() == pytest.approx(1., abs=1e-13)
    gradient = torch.autograd.grad(torch.exp(log_prob).sum(), scores)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-13, rtol=0)
    torch.testing.assert_close(plackett_luce_log_prob(scores + 900., orders), log_prob, atol=1e-12, rtol=0)


def test_pl_two_item_log_probability_has_no_trajectory_length_scaling():
    scores = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
    actual = plackett_luce_log_prob(scores, torch.tensor([0, 1]))
    expected = F.logsigmoid(scores[0] - scores[1])
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, scores)[0]
    expected_gradient = torch.tensor([1., -1.], dtype=torch.float64) * torch.sigmoid(scores[1] - scores[0])
    torch.testing.assert_close(gradient, expected_gradient)


def test_gumbel_sampler_matches_full_pl_law_and_is_detached():
    scores = torch.tensor([.6, -.8, .1], dtype=torch.float64, requires_grad=True)
    orders = torch.tensor(list(itertools.permutations(range(3))), dtype=torch.long)
    expected = plackett_luce_log_prob(scores, orders).exp().detach()
    sampled = sample_plackett_luce(scores, 50000, torch.Generator().manual_seed(919))
    assert not sampled.requires_grad
    for order, probability in zip(orders, expected):
        observed = (sampled == order).all(dim=1).to(torch.float64).mean()
        assert abs(observed.item() - probability.item()) < .008


def test_ap_rewards_match_known_rankings_and_edge_labels():
    labels = torch.tensor([1, 0, 1, 0])
    orders = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2], [3, 2, 1, 0]])
    expected = torch.tensor([5 / 6, 5 / 12, .5], dtype=torch.float64)
    torch.testing.assert_close(permutation_ap_rewards(labels, orders), expected)
    assert permutation_ap_rewards(torch.zeros(4), orders).tolist() == [0., 0., 0.]
    assert permutation_ap_rewards(torch.ones(4), orders).tolist() == [1., 1., 1.]


@pytest.mark.parametrize("group_size", [2, 3])
def test_exact_enumeration_rloo_gradient_matches_expected_ap_gradient(group_size):
    scores = torch.tensor([.4, -.3, .8], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 0, 0])
    orders = torch.tensor(list(itertools.permutations(range(3))), dtype=torch.long)
    log_prob = plackett_luce_log_prob(scores, orders)
    rewards = permutation_ap_rewards(labels, orders)
    expected_reward = (log_prob.exp() * rewards).sum()
    exact_gradient = torch.autograd.grad(expected_reward, scores, retain_graph=True)[0]
    expected_estimator = scores.sum() * 0
    for group in itertools.product(range(len(orders)), repeat=group_size):
        indices = torch.tensor(group)
        # Sampling weights are detached: differentiate the score-function
        # estimator, not the expectation's distribution a second time.
        probability = log_prob[indices].sum().exp().detach()
        advantage = group_advantages(rewards[indices], "rloo_ap")
        expected_estimator = expected_estimator - probability * (advantage * log_prob[indices]).mean()
    estimator_gradient = torch.autograd.grad(expected_estimator, scores)[0]
    torch.testing.assert_close(estimator_gradient, -exact_gradient, atol=1e-12, rtol=1e-12)


def test_macro_bce_uses_disease_denominators_and_masks_prior_gradient():
    logits = torch.tensor([[.2, 0.], [100., 1.], [-100., -1.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1, 0], [0, 0], [0, 1]])
    eligible = torch.tensor([[1, 1], [0, 1], [0, 1]], dtype=torch.bool)
    loss = macro_bce_logits(logits, labels, eligible)
    expected = .5 * (F.softplus(-logits[0, 0]) + F.binary_cross_entropy_with_logits(
        logits[:, 1], labels[:, 1].to(torch.float64)))
    torch.testing.assert_close(loss, expected)
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad[1:, 0].tolist() == [0., 0.]


@pytest.mark.parametrize("method", METHODS)
def test_empty_positive_or_negative_diseases_have_zero_rank_contribution_with_fixed_k(method):
    first = torch.tensor([.8, -.2, .1], dtype=torch.float64)
    logits = torch.stack([first, first, first], dim=1).requires_grad_()
    labels = torch.tensor([[1, 0, 1], [0, 0, 1], [0, 0, 1]])
    eligible = torch.ones_like(labels, dtype=torch.bool)
    full, stats = rank_loss(logits, labels, eligible, method, generator=torch.Generator().manual_seed(42))
    one, _ = rank_loss(logits[:, :1], labels[:, :1], eligible[:, :1], method,
                       generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(full, one / 3)
    grad = torch.autograd.grad(full, logits)[0]
    assert stats["active_conditions"] == 1
    torch.testing.assert_close(grad[:, 1:], torch.zeros_like(grad[:, 1:]))


@pytest.mark.parametrize("method", ["ranknet", "smoothap"])
def test_global_chunking_preserves_loss_and_gradient(method):
    labels = torch.tensor([[1], [0], [1], [0], [0], [1], [0]])
    eligible = torch.ones_like(labels, dtype=torch.bool)
    outputs = []
    for chunk_size in (1, 2, 128):
        logits = torch.tensor([[.3], [-.2], [.7], [.1], [-.8], [.6], [.4]],
                              dtype=torch.float64, requires_grad=True)
        value, _ = rank_loss(logits, labels, eligible, method, temperature=.6, chunk_size=chunk_size)
        outputs.append((value.detach(), torch.autograd.grad(value, logits)[0]))
    for output in outputs[1:]:
        torch.testing.assert_close(output[0], outputs[0][0], atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(output[1], outputs[0][1], atol=1e-13, rtol=1e-13)


def test_smooth_ap_low_temperature_converges_to_ap_without_ties():
    logits = torch.tensor([[1.2], [1.], [-.4], [.4]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1], [0], [1], [0]])
    actual, _ = rank_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool), "smoothap", temperature=1e-4)
    exact = permutation_ap_rewards(labels[:, 0], torch.argsort(logits[:, 0], descending=True))
    assert actual.item() == pytest.approx(1 - exact.item(), abs=1e-12)
    assert actual.item() == pytest.approx(.25)


def test_smooth_ap_autograd_matches_finite_differences():
    logits = torch.tensor([[.4], [-.2], [.6]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1], [0], [1]])
    eligible = torch.ones_like(labels, dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda value: rank_loss(value, labels, eligible, "smoothap", temperature=.7)[0],
                                   (logits,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_listnet_is_uniform_positive_top_one_cross_entropy():
    logits = torch.tensor([[.7], [-.1], [.3]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1], [0], [1]])
    value, _ = rank_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool), "listnet", temperature=.8)
    expected = -F.log_softmax(logits[:, 0] / .8, dim=0)[labels[:, 0].bool()].mean()
    torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("method", METHODS)
def test_shift_invariance_and_ineligible_cells_do_not_affect_ranking_or_gradient(method):
    base = torch.tensor([[.3, -.5], [-.1, .8], [.9, -.3], [.1, .2]], dtype=torch.float64)
    labels = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    eligible = torch.tensor([[1, 1], [1, 0], [1, 1], [0, 1]], dtype=torch.bool)
    outputs = []
    for change in (False, True):
        values = base.clone()
        if change:
            values = values + torch.tensor([7., -2.], dtype=torch.float64)
            values[~eligible] += 1000.
        values.requires_grad_()
        loss, stats = rank_loss(values, labels, eligible, method, temperature=torch.tensor([.7, 1.4]),
                                generator=torch.Generator().manual_seed(18))
        gradient = torch.autograd.grad(loss, values)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert torch.all(gradient[~eligible] == 0)
        torch.testing.assert_close(gradient.sum(dim=0), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
        json.dumps(stats, allow_nan=False)
        outputs.append((loss.detach(), gradient))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=1e-12, rtol=1e-12)


def test_two_policy_methods_use_identical_random_stream_and_leave_global_rng_unchanged():
    logits = torch.tensor([[.3, -.1], [.9, -.4], [-.1, .7], [.4, .1]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    eligible = torch.ones_like(labels, dtype=torch.bool)
    state = torch.get_rng_state().clone()
    generators = [torch.Generator().manual_seed(15), torch.Generator().manual_seed(15)]
    records = []
    for method, generator in zip(("rloo_ap", "grpo_ap"), generators):
        _, stats = rank_loss(logits, labels, eligible, method, generator=generator, group_size=8)
        records.append(stats)
    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    assert torch.equal(state, torch.get_rng_state())
    assert records[0]["reward_mean"] == records[1]["reward_mean"]
    assert records[0]["reward_std"] == records[1]["reward_std"]
    assert records[0]["sample_draws"] == 64
    assert records[0]["n_trajectories"] == 16


@pytest.mark.parametrize("method", ["rloo_ap", "grpo_ap"])
def test_zero_reward_variance_gives_finite_zero_policy_gradient(method):
    logits = torch.tensor([[1e6], [-1e6]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1], [0]])
    loss, stats = rank_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool), method,
                            generator=torch.Generator().manual_seed(11), group_size=8)
    grad = torch.autograd.grad(loss, logits)[0]
    assert loss.item() == 0 and torch.all(grad == 0)
    assert stats["zero_var_fraction"] == 1 and stats["reward_std"] == 0
    json.dumps(stats, allow_nan=False)


def test_control_is_zero_rank_component_and_no_eligible_bce_is_differentiable_zero():
    logits = torch.tensor([[.2], [.3]], requires_grad=True)
    labels = torch.tensor([[1], [0]])
    eligible = torch.ones_like(labels, dtype=torch.bool)
    rank, stats = rank_loss(logits, labels, eligible, "macro_bce")
    assert rank.item() == 0 and stats["active_conditions"] == 0
    assert macro_bce_logits(logits, labels, eligible).item() > 0
    empty = macro_bce_logits(logits, labels, ~eligible)
    assert empty.item() == 0
    assert torch.all(torch.autograd.grad(empty, logits)[0] == 0)


def test_invalid_inputs_are_rejected():
    logits = torch.tensor([[.2], [.3]], requires_grad=True)
    labels = torch.tensor([[1], [0]])
    eligible = torch.ones_like(labels, dtype=torch.bool)
    for kw in (dict(temperature=0), dict(chunk_size=0), dict(temperature=torch.tensor([1.], requires_grad=True))):
        with pytest.raises(ValueError):
            rank_loss(logits, labels, eligible, "ranknet", **kw)
    with pytest.raises(ValueError):
        rank_loss(logits, labels, eligible, "rloo_ap", group_size=1)
    with pytest.raises(ValueError):
        rank_loss(logits, labels, eligible, "invented")
    with pytest.raises(ValueError):
        macro_bce_logits(logits, labels + 2, eligible)
    with pytest.raises(ValueError):
        plackett_luce_log_prob(logits[:, 0], torch.tensor([0, 0]))
    with pytest.raises(ValueError):
        permutation_ap_rewards(labels[:, 0], torch.tensor([0, 0]))
