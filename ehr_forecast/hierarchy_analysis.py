"""Paired validation comparisons of single-model quality across fixed seeds.

The estimand is the mean of paired per-seed metric differences, not the metric
of a probability ensemble. One patient bootstrap draw is applied jointly to
every condition, seed, and model family. Random seeds are kept fixed rather
than being resampled; patient uncertainty and fitting variability are reported
separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .experiment_analysis import (
    UNCERTAINTY_NOTE, _bootstrap_interval, _group_mask, _score_orders, _weighted_macro_metrics,
)
from .metrics import _validate_target_codes, evaluate_predictions, validate_probability_matrix


def _validate_seed_predictions(predictions, labels, codes, name):
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError(f"{name} must be a nonempty mapping from integer seed to probability matrix")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in predictions):
        raise ValueError("Model seeds must be nonnegative integers")
    return {seed: validate_probability_matrix(scores, n_patients=len(labels), n_targets=len(codes))
            for seed, scores in predictions.items()}


def _seed_summary(values):
    defined = [float(value) for value in values if value is not None]
    return {"mean": float(np.mean(defined)) if defined else None,
            "sample_std": float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
            "min": min(defined) if defined else None,
            "max": max(defined) if defined else None,
            "n_defined": len(defined)}


def _wins(values):
    defined = [float(value) for value in values if value is not None]
    return {"wins": sum(value > 1e-12 for value in defined),
            "ties": sum(abs(value) <= 1e-12 for value in defined),
            "losses": sum(value < -1e-12 for value in defined),
            "n_defined": len(defined), "tie_tolerance": 1e-12}


def paired_seed_bootstrap(
    y_true: Any,
    predictions_a: Mapping[int, Any],
    predictions_b: Mapping[int, Any],
    target_codes: Sequence[str],
    mask: Any | None = None,
    n_bootstrap: int = 500,
    seed: int = 2026,
    confidence: float = 0.95,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Mean paired single-fit B-A differences with patient-percentile intervals.

    Matrices must already share the same patient and target ordering. ``mask``
    may select a fixed boolean patient subgroup, such as the original >255-event
    long-history cohort. Group membership must be defined without model outcomes.

    Each resample calculates mAP separately for each seed and then averages the
    B-A differences. It never averages seed probabilities. All target classes
    stay in mAP (AP=0 when a resample has no positives); AUROC excludes undefined
    classes identically for both models and all seeds on that resample.
    """
    codes = _validate_target_codes(target_codes)
    labels = np.asarray(y_true)
    # Production validation includes binary labels, nonempty rows and exact C.
    evaluate_predictions(labels, np.zeros_like(labels, dtype=float), codes)
    labels = np.asarray(labels, dtype=np.float64)
    scores_a = _validate_seed_predictions(predictions_a, labels, codes, "predictions_a")
    scores_b = _validate_seed_predictions(predictions_b, labels, codes, "predictions_b")
    seeds = sorted(scores_a)
    if set(scores_a) != set(scores_b):
        raise ValueError("Both families must contain exactly the same paired seeds")
    if expected_seeds is not None:
        expected = list(expected_seeds)
        if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in expected)
                or len(expected) != len(set(expected)) or set(expected) != set(seeds)):
            raise ValueError("Observed seeds do not match the unique expected seeds")
    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    n_reference = len(labels)
    if mask is not None:
        selected = np.asarray(mask)
        if selected.dtype.kind != "b" or selected.shape != (n_reference,):
            raise ValueError("mask must be boolean with one entry per reference patient")
        if not selected.any():
            raise ValueError("Selected patient subgroup is empty")
        labels = labels[selected]
        scores_a = {key: value[selected] for key, value in scores_a.items()}
        scores_b = {key: value[selected] for key, value in scores_b.items()}

    per_seed = []
    orders_a, orders_b = {}, {}
    for model_seed in seeds:
        metric_a = evaluate_predictions(labels, scores_a[model_seed], codes)
        metric_b = evaluate_predictions(labels, scores_b[model_seed], codes)
        point_a = {key: metric_a[key] for key in ("mAP", "macro_auroc", "n_auroc_valid")}
        point_b = {key: metric_b[key] for key in ("mAP", "macro_auroc", "n_auroc_valid")}
        per_seed.append({"seed": model_seed, "a": point_a, "b": point_b,
                         "delta_mAP": metric_b["mAP"] - metric_a["mAP"],
                         "delta_macro_auroc": metric_b["macro_auroc"] - metric_a["macro_auroc"]
                         if metric_a["macro_auroc"] is not None else None})
        orders_a[model_seed] = _score_orders(scores_a[model_seed])
        orders_b[model_seed] = _score_orders(scores_b[model_seed])
    map_summary = _seed_summary([row["delta_mAP"] for row in per_seed])
    auc_summary = _seed_summary([row["delta_macro_auroc"] for row in per_seed])
    rng = np.random.default_rng(seed)
    map_deltas, auc_deltas, valid_counts = [], [], []
    for _ in range(n_bootstrap):
        # The same multiplicity weights are used for all seeds/families.
        indices = rng.integers(0, len(labels), size=len(labels))
        weights = np.bincount(indices, minlength=len(labels)).astype(np.float64)
        per_seed_maps, per_seed_aucs = [], []
        for model_seed in seeds:
            a = _weighted_macro_metrics(labels, orders_a[model_seed], weights)
            b = _weighted_macro_metrics(labels, orders_b[model_seed], weights)
            per_seed_maps.append(b["mAP"] - a["mAP"])
            if a["macro_auroc"] is not None:
                per_seed_aucs.append(b["macro_auroc"] - a["macro_auroc"])
        map_deltas.append(float(np.mean(per_seed_maps)))
        auc_deltas.append(float(np.mean(per_seed_aucs)) if per_seed_aucs else None)
        valid_counts.append(a["n_auroc_valid"])
    return {
        "estimand": "mean across paired seeds of SINGLE-MODEL metric B minus A",
        "uses_probability_ensemble": False,
        "difference_direction": "B minus A", "seeds": seeds, "n_seeds": len(seeds),
        "n_reference_patients": n_reference, "n_patients": len(labels), "n_targets": len(codes),
        "positive_instances": int(labels.sum()),
        "n_positive_by_target": labels.sum(axis=0).astype(int).tolist(),
        "per_seed": per_seed,
        "seed_deltas": [{"seed": row["seed"], "mAP": row["delta_mAP"],
                         "macro_auroc": row["delta_macro_auroc"]} for row in per_seed],
        "mean_paired_delta": {"mAP": map_summary["mean"], "macro_auroc": auc_summary["mean"]},
        "wins": {"mAP": _wins([row["delta_mAP"] for row in per_seed]),
                 "macro_auroc": _wins([row["delta_macro_auroc"] for row in per_seed])},
        "mean_single_model_metrics": {
            side: {metric: _seed_summary([row[side][metric] for row in per_seed])
                   for metric in ("mAP", "macro_auroc")} for side in ("a", "b")
        },
        "between_seed_deltas": {"mAP": map_summary, "macro_auroc": auc_summary},
        "delta_mAP": _bootstrap_interval(map_deltas, map_summary["mean"], confidence),
        "delta_macro_auroc": _bootstrap_interval(auc_deltas, auc_summary["mean"], confidence),
        "bootstrap_n_auroc_valid": {"min": min(valid_counts), "max": max(valid_counts),
                                     "mean": float(np.mean(valid_counts))},
        "n_bootstrap": n_bootstrap, "bootstrap_seed": seed, "confidence_level": confidence,
        "resampling_unit": "one shared patient resample across all conditions, families, and fixed seeds",
        "metric_policy": "All target APs including no-positive AP=0; AUROC eligibility depends only on labels and is shared across models/seeds.",
        "uncertainty_note": UNCERTAINTY_NOTE + " Seeds are not resampled; this interval is conditional on the listed fitted seeds."
    }


def paired_seed_group_comparisons(
    y_true: Any,
    predictions_a: Mapping[int, Any],
    predictions_b: Mapping[int, Any],
    target_codes: Sequence[str],
    groups: Mapping[str, Any],
    patient_ids: Sequence[str] | None = None,
    *,
    n_bootstrap: int = 500,
    seed: int = 2026,
    confidence: float = 0.95,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Apply the same paired-single-model estimand overall and in fixed groups.

    Groups accept boolean masks or unique patient IDs when ``patient_ids`` is
    supplied. An empty group returns ``None`` rather than fabricating scores.
    No group is used for selecting a checkpoint or seed by this function.
    """
    if not isinstance(groups, Mapping):
        raise ValueError("groups must map names to fixed patient masks or IDs")
    labels = np.asarray(y_true)
    if labels.ndim != 2:
        raise ValueError("Labels must have shape [patients, target codes]")
    ids = list(patient_ids) if patient_ids is not None else None
    if ids is not None and (len(ids) != len(labels) or len(set(ids)) != len(ids)):
        raise ValueError("patient_ids must uniquely match label rows")
    controls = dict(n_bootstrap=n_bootstrap, seed=seed, confidence=confidence,
                    expected_seeds=expected_seeds)
    overall = paired_seed_bootstrap(labels, predictions_a, predictions_b, target_codes, **controls)
    comparisons = {}
    counts = {}
    for name, values in groups.items():
        mask = _group_mask(values, len(labels), ids)
        counts[str(name)] = int(mask.sum())
        comparisons[str(name)] = paired_seed_bootstrap(labels, predictions_a, predictions_b, target_codes,
                                                       mask=mask, **controls) if mask.any() else None
    return {"overall": overall, "groups": comparisons, "group_patient_counts": counts,
            "uses_probability_ensemble": False,
            "group_comparison_note": "Compare model differences within the same fixed group; group prevalences and class supports differ."}
