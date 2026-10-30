"""Reference-scale ranking objectives, isolated from the frozen first study.

For one disease, write its eligible raw scores as s, their current mean/std as
mu/sigma, and the fixed theta0 std as sigma0. The residual formula equals
z = mu + sigma0*(s-mu)/sigma. For s'=a*s+b, a>0, it gives
z'=z+(a-1)*mu+b. Thus z changes only by a common offset: the complete
Plackett--Luce distribution and difference-based SmoothAP are unchanged at a
fixed temperature. Full differentiation through current mu/sigma is essential.

At theta0, when sigma0 is computed with the same dtype/population-std operation,
the ratio is exactly1 and the residual is exactly zero: forward policy scores
are identical, while their derivatives deliberately project out score scaling.
No training, feature extraction, or temperature fitting occurs in this module.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .objectives import rank_loss as _original_rank_loss


def _normalized_parts(logits: Tensor, eligible: Tensor, reference_std: Tensor, eps: float):
    if logits.ndim != 2 or not logits.shape[0] or not logits.shape[1] or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty floating [patients, diseases] tensor")
    if eligible.shape != logits.shape or eligible.device != logits.device:
        raise ValueError("eligible must have the same shape/device as logits")
    if not bool(((eligible == 0) | (eligible == 1)).all()):
        raise ValueError("eligible must be binary")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    mask = eligible.bool()
    counts = mask.sum(dim=0)
    if not bool((counts >= 2).all()):
        raise ValueError("Each disease requires at least two eligible training patients")
    reference = torch.as_tensor(reference_std, dtype=logits.dtype, device=logits.device).detach()
    if reference.shape != (logits.shape[1],) or not bool(torch.isfinite(reference).all()):
        raise ValueError("reference_std must be finite with shape [diseases]")
    if not bool((reference > eps).all()):
        raise ValueError("Degenerate reference_std: every disease must have std>eps")
    # Match the exact per-column operation used to define theta0 reference_std;
    # do not substitute a different vectorized variance formula or detach these.
    means = torch.stack([logits[:, c][mask[:, c]].mean() for c in range(logits.shape[1])])
    current = torch.stack([logits[:, c][mask[:, c]].std(correction=0) for c in range(logits.shape[1])])
    if not bool(torch.isfinite(current).all()) or not bool((current > eps).all()):
        raise ValueError("Degenerate current eligible score std: every disease must have std>eps")
    ratio = reference / current.clamp_min(eps)
    # The guard above keeps clamp inactive. It must not silently replace an
    # ill-defined scale-invariant objective with a clipped different objective.
    normalized = logits + (logits - means.unsqueeze(0)) * (ratio.unsqueeze(0) - 1)
    return normalized, current, reference, ratio


def normalize_scores(logits: Tensor, eligible: Tensor, reference_std: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize only for ranking; current eligible moments remain differentiable.

    reference_std is always detached. Excluded cells do not define moments;
    their returned values are unused, and the rank wrapper masks them before
    any objective, ensuring zero rank gradient for every excluded cell.
    Degenerate reference/current stds or insufficient eligible rows raise.
    """
    return _normalized_parts(logits, eligible, reference_std, eps)[0]


def rank_loss(
    logits: Tensor,
    labels: Tensor,
    eligible: Tensor,
    method: str,
    temperature: Tensor | float = 1.,
    generator: torch.Generator | None = None,
    group_size: int = 8,
    chunk_size: int = 128,
    *,
    reference_std: Tensor,
    eps: float = 1e-6,
    advantage_eps: float = 1e-8,
):
    """SI SmoothAP/RLOO/GRPO rank component; BCE itself must use raw logits.

    Old argument positions are retained; reference_std is a required keyword.
    ``bce``/``macro_bce``/``none`` return the original zero rank component
    without normalizing the raw BCE arm. No lambda/temperature adjustment occurs.
    """
    name = method.lower().replace("-", "_").strip()
    if name in {"bce", "macro_bce", "macrobce", "none"}:
        loss, stats = _original_rank_loss(logits, labels, eligible, "bce", temperature,
            generator=generator, group_size=group_size, chunk_size=chunk_size, eps=advantage_eps)
        stats.update({"method": name, "base_method": "bce", "score_normalization": None,
                      "score_normalization_full_derivatives": False})
        return loss, stats
    methods = {"smoothap_si": "smoothap", "smooth_ap_si": "smoothap",
               "rloo_ap_si": "rloo_ap", "grpo_ap_si": "grpo_ap"}
    if name not in methods:
        raise ValueError("method must be bce, smoothap_si, rloo_ap_si or grpo_ap_si")
    normalized, current, reference, ratio = _normalized_parts(logits, eligible, reference_std, eps)
    loss, stats = _original_rank_loss(normalized, labels, eligible, methods[name], temperature,
        generator=generator, group_size=group_size, chunk_size=chunk_size, eps=advantage_eps)
    stats.update({"method": name, "base_method": methods[name],
                  "score_normalization": "reference train eligible std, residual form",
                  "score_normalization_full_derivatives": True,
                  "raw_std_min": float(current.detach().min().cpu()),
                  "reference_std_min": float(reference.min().cpu()),
                  "reference_std_max": float(reference.max().cpu()),
                  "scale_ratio_min": float(ratio.detach().min().cpu()),
                  "scale_ratio_max": float(ratio.detach().max().cpu())})
    return loss, stats
