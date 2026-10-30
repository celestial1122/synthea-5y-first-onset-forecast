"""Full-training-pool objectives for an isolated frozen-encoder head study.

Every ranking objective normalizes *within* a disease and then over all K
diseases. There are no comparisons between different diseases. Prior/ineligible
cells are removed before ranking and receive zero gradient.

The two policy objectives are on-policy, group-relative REINFORCE adaptations:
fresh Plackett--Luce trajectories at every call, no PPO clipping, no rollout
reuse, no old policy, and no claim of implementing the original GRPO algorithm.
RLOO is an unbiased score-function estimator of expected sampled AP. Dividing
by a reward-dependent group standard deviation (the GRPO-inspired variant)
changes the estimator and is not claimed to have that unbiasedness property.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def _inputs(logits: Tensor, labels: Tensor, eligible: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2 or not logits.shape[0] or not logits.shape[1] or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty floating [patients, diseases] tensor")
    if labels.shape != logits.shape or eligible.shape != logits.shape:
        raise ValueError("labels and eligible must have the same shape as logits")
    if labels.device != logits.device or eligible.device != logits.device:
        raise ValueError("logits, labels and eligible must be on the same device")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("logits/labels must be finite")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be binary")
    if not bool(((eligible == 0) | (eligible == 1)).all()):
        raise ValueError("eligible must be binary")
    return labels.bool(), eligible.bool()


def _temperatures(temperature: Tensor | float, logits: Tensor) -> Tensor:
    values = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
    if values.requires_grad:
        raise ValueError("temperature is fixed and must not require gradients")
    if values.ndim == 0:
        values = values.expand(logits.shape[1])
    if values.shape != (logits.shape[1],) or not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("temperature must be positive, finite and scalar or [diseases]")
    return values


def macro_bce_logits(logits: Tensor, labels: Tensor, eligible: Tensor) -> Tensor:
    """Mean disease BCE; each disease first averages its own eligible patients.

    A disease with no eligible patients contributes differentiable zero and is
    still included in the fixed K denominator. No class-positive reweighting.
    """
    truth, mask = _inputs(logits, labels, eligible)
    losses = F.binary_cross_entropy_with_logits(logits, truth.to(logits.dtype), reduction="none")
    counts = mask.sum(dim=0).clamp_min(1)
    return ((losses * mask).sum(dim=0) / counts).mean()


def sample_plackett_luce(scores: Tensor, group_size: int, generator: torch.Generator | None = None) -> Tensor:
    """Sample [G,N] permutations via detached Gumbel perturbations.

    ``scores`` are the already temperature-scaled scalar item scores. The
    supplied Generator must match their device. Only this function consumes RNG.
    """
    if scores.ndim != 1 or not scores.numel() or not scores.is_floating_point():
        raise ValueError("scores must be a nonempty floating vector")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    uniform = torch.rand((group_size, scores.numel()), device=scores.device,
                         dtype=scores.dtype, generator=generator)
    # rand is strictly less than1. Only guard its possible zero endpoint.
    gumbel = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(scores.dtype).tiny)))
    return torch.argsort(scores.detach().unsqueeze(0) + gumbel, dim=1,
                         descending=True, stable=True)


def plackett_luce_log_prob(scores: Tensor, permutations: Tensor, *, validate: bool = True) -> Tensor:
    """Differentiable *full trajectory* PL log-probability, with no N scaling.

    A single [N] permutation returns a scalar; [G,N] returns [G].
    """
    if scores.ndim != 1 or not scores.numel() or not scores.is_floating_point():
        raise ValueError("scores must be a nonempty floating vector")
    single = permutations.ndim == 1
    order = permutations.unsqueeze(0) if single else permutations
    if order.ndim != 2 or order.shape[1] != scores.numel() or order.dtype != torch.long:
        raise ValueError("permutations must be long [N] or [G,N]")
    if order.device != scores.device:
        raise ValueError("permutations and scores must share a device")
    if validate:
        expected = torch.arange(scores.numel(), device=scores.device).expand_as(order)
        if not bool(torch.equal(order.sort(dim=1).values, expected)):
            raise ValueError("Every row must be a permutation of0..N-1")
    ordered = scores[order]
    remaining_log_mass = torch.logcumsumexp(ordered.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
    result = (ordered - remaining_log_mass).sum(dim=-1)
    return result[0] if single else result


def permutation_ap_rewards(labels: Tensor, permutations: Tensor, *, validate: bool = True) -> Tensor:
    """Exact ordinary AP of each strict sampled permutation; no-positive AP=0."""
    if labels.ndim != 1 or not labels.numel() or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be a nonempty binary vector")
    single = permutations.ndim == 1
    order = permutations.unsqueeze(0) if single else permutations
    if order.ndim != 2 or order.shape[1] != labels.numel() or order.dtype != torch.long:
        raise ValueError("permutations must be long [N] or [G,N]")
    if labels.device != order.device:
        raise ValueError("labels/permutations must share a device")
    if validate:
        expected = torch.arange(labels.numel(), device=labels.device).expand_as(order)
        if not bool(torch.equal(order.sort(dim=1).values, expected)):
            raise ValueError("Every row must be a permutation of0..N-1")
    # Use double rewards for stable small reward differences; trajectories are
    # discrete and rewards are deliberately detached from policy gradients.
    # CUDA's floating cumsum is not deterministic in the project's PyTorch
    # build. Relevant-item counts are integers, so accumulate exactly as int64
    # before converting to double for precision/AP arithmetic.
    ordered = labels[order].to(torch.int64)
    positions = torch.arange(1, labels.numel() + 1, device=labels.device, dtype=torch.float64)
    precisions = ordered.cumsum(dim=-1, dtype=torch.int64).to(torch.float64) / positions
    rewards = (precisions * ordered).sum(dim=-1) / labels.sum().clamp_min(1)
    return rewards[0] if single else rewards


def group_advantages(rewards: Tensor, method: str, eps: float = 1e-8) -> Tensor:
    """Detached RLOO or normalized group-relative advantages for one disease."""
    if rewards.ndim != 1 or rewards.numel() < 2 or not bool(torch.isfinite(rewards).all()):
        raise ValueError("At least two finite group rewards are required")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    values = rewards.detach()
    if method in {"rloo", "rloo_ap"}:
        return values - (values.sum() - values) / (values.numel() - 1)
    if method in {"grpo", "grpo_ap"}:
        centered = values - values.mean()
        return centered / (centered.square().mean().sqrt() + eps)
    raise ValueError("method must be rloo_ap or grpo_ap")


def _ranknet(scores: Tensor, truth: Tensor, chunk_size: int) -> Tensor:
    positive, negative = scores[truth], scores[~truth]
    result = scores.sum() * 0
    for part in positive.split(chunk_size):
        result = result + F.softplus(negative.unsqueeze(0) - part.unsqueeze(1)).sum()
    return result / (positive.numel() * negative.numel())


def _smooth_ap(scores: Tensor, truth: Tensor, chunk_size: int) -> Tensor:
    positive = scores[truth]
    precision_sum = scores.sum() * 0
    for part in positive.split(chunk_size):
        greater = torch.sigmoid(scores.unsqueeze(0) - part.unsqueeze(1))
        # The self-comparison contributes sigmoid(0)=.5 to each sum; exclude
        # that item and add its guaranteed rank/relevant count of1 instead.
        soft_rank = .5 + greater.sum(dim=1)
        soft_positive_rank = .5 + (greater * truth).sum(dim=1)
        precision_sum = precision_sum + (soft_positive_rank / soft_rank).sum()
    return 1 - precision_sum / positive.numel()


def rank_loss(
    logits: Tensor,
    labels: Tensor,
    eligible: Tensor,
    method: str,
    temperature: Tensor | float = 1.,
    generator: torch.Generator | None = None,
    group_size: int = 8,
    chunk_size: int = 128,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, Any]]:
    """A macro-normalized full-pool ranking loss and JSON-safe diagnostics.

    Methods: ``ranknet``, ``listnet``, ``smoothap``, ``rloo_ap``, ``grpo_ap``.
    ``bce``/``macro_bce``/``none`` return a zero *rank component*; call
    :func:`macro_bce_logits` for the actual BCE objective.

    Diseases missing an eligible positive or negative have constant AP; they
    contribute differentiable zero, still divided by all K diseases. Chunking
    only bounds temporary matrices and never changes the global denominators.
    """
    truth, mask = _inputs(logits, labels, eligible)
    temperatures = _temperatures(temperature, logits)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    name = method.lower().replace("-", "_").strip()
    aliases = {"smooth_ap": "smoothap", "rloo": "rloo_ap", "grpo": "grpo_ap", "macrobce": "macro_bce"}
    name = aliases.get(name, name)
    allowed = {"ranknet", "listnet", "smoothap", "rloo_ap", "grpo_ap", "bce", "macro_bce", "none"}
    if name not in allowed:
        raise ValueError(f"Unknown rank method: {method}")
    if name in {"rloo_ap", "grpo_ap"} and (not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 2):
        raise ValueError("Policy objectives require group_size>=2")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    result = logits.sum() * 0
    rewards_record, log_prob_record = [], []
    positive_counts = (truth & mask).sum(dim=0).detach().cpu().tolist()
    negative_counts = (~truth & mask).sum(dim=0).detach().cpu().tolist()
    active = 0
    scored_items, pair_count = 0, 0
    if name not in {"bce", "macro_bce", "none"}:
        for c, (n_positive, n_negative) in enumerate(zip(positive_counts, negative_counts)):
            if not n_positive or not n_negative:
                continue
            active += 1
            scored_items += n_positive + n_negative
            pair_count += n_positive * n_negative
            scores = logits[:, c][mask[:, c]] / temperatures[c]
            observed = truth[:, c][mask[:, c]]
            if name == "ranknet":
                component = _ranknet(scores, observed, chunk_size)
            elif name == "listnet":
                # Uniform binary-positive top-one target, not an AP loss.
                component = torch.logsumexp(scores, dim=0) - scores[observed].mean()
            elif name == "smoothap":
                component = _smooth_ap(scores, observed, chunk_size)
            else:
                permutations = sample_plackett_luce(scores, group_size, generator)
                log_prob = plackett_luce_log_prob(scores, permutations, validate=False)
                reward = permutation_ap_rewards(observed, permutations, validate=False)
                advantages = group_advantages(reward, name, eps)
                # No division by trajectory length. Casting detached advantages
                # preserves the head dtype without differentiating reward stats.
                component = -(advantages.to(log_prob.dtype) * log_prob).mean()
                rewards_record.append(reward.detach())
                log_prob_record.append(log_prob.detach().to(torch.float64))
            result = result + component / logits.shape[1]
    policy = name in {"rloo_ap", "grpo_ap"}
    if rewards_record:
        rewards = torch.stack(rewards_record)
        reward_std = rewards.std(dim=1, correction=0)
        reward_mean = float(rewards.mean().cpu())
        mean_std = float(reward_std.mean().cpu())
        zero_fraction = float((reward_std == 0).to(torch.float64).mean().cpu())
        mean_log_prob = float(torch.stack(log_prob_record).mean().cpu())
    else:
        reward_mean = mean_std = zero_fraction = mean_log_prob = None
    stats = {"method": name, "active_conditions": active, "total_conditions": logits.shape[1],
             "inactive_conditions": logits.shape[1] - active, "normalization": "within disease then fixed K",
             "reward_mean": reward_mean, "reward_std": mean_std, "zero_var_fraction": zero_fraction,
             "mean_trajectory_log_prob": mean_log_prob, "group_size": group_size if policy else None,
             "n_trajectories": active * group_size if policy else 0,
             "sample_draws": scored_items * group_size if policy else 0,
             "n_rank_pairs": pair_count, "scored_items": scored_items,
             "policy_description": "fresh on-policy group-relative REINFORCE adaptation, no PPO/reuse/N scaling" if policy else None}
    return result, stats
