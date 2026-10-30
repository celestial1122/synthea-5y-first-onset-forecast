"""Validation-only experiment comparisons with explicit uncertainty semantics.

Patient bootstrapping resamples rows jointly across conditions and models.  It
describes sampling uncertainty conditional on these fitted, selected models and
this validation cohort; it does not undo early stopping/model selection or
estimate hidden-test generalization.  Random-seed variation is reported
separately and is never reduced to a best-seed score.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import _validate_target_codes, evaluate_predictions, validate_probability_matrix
from .submission import validate_predictions


UNCERTAINTY_NOTE = (
    "Patient bootstrap is conditional on this fixed validation cohort and the fitted models. "
    "It does not account for using validation to select epochs, architectures, or hyperparameters; "
    "it is not an unbiased test-performance interval or clinical-generalization claim. "
    "Between-seed variation and between-patient sampling uncertainty are distinct."
)


def load_aligned_predictions(
    path: str | Path, patient_ids: Sequence[str], target_codes: Sequence[str]
) -> np.ndarray:
    """Validate exact columns/coverage, then align arbitrary CSV rows to official IDs."""
    source = Path(path)
    ids, codes = list(patient_ids), _validate_target_codes(target_codes)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Reference patient IDs must be unique and nonempty")
    validate_predictions(source, codes, ids)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        by_id = {row[0]: [float(value) for value in row[1:]] for row in reader}
    return np.asarray([by_id[patient] for patient in ids], dtype=np.float64)


def _validation_reference(prepared_dir: str | Path) -> dict[str, Any]:
    prepared = Path(prepared_dir)
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    codes = _validate_target_codes(metadata["target_codes"])
    with np.load(prepared / "val.npz", allow_pickle=False) as arrays:
        ids = arrays["patient_ids"].astype(str).tolist()
        labels = np.asarray(arrays["labels"], dtype=np.float64)
    # Reuse the production label/shape validation and rare-target policy.
    evaluate_predictions(labels, np.zeros_like(labels), codes)
    if len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Validation reference has inconsistent/duplicate patient IDs")
    return {"patient_ids": ids, "labels": labels, "target_codes": codes,
            "target_descriptions": metadata.get("target_descriptions")}


def _load_runs(run_dirs, reference, expected_seeds=None):
    directories = [Path(path) for path in run_dirs]
    if not directories or len({path.resolve() for path in directories}) != len(directories):
        raise ValueError("Provide nonempty, unique run directories")
    runs, seen_seeds, family_signature = [], set(), None
    for directory in directories:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["target_codes"] != reference["target_codes"]:
            raise ValueError(f"Run target ordering differs from reference: {directory}")
        ignored = {"seed", "name", "output_dir", "prepared_dir", "wandb_mode", "wandb_project", "wandb_entity", "device"}
        signature = json.dumps({"training": {key: value for key, value in config.items() if key not in ignored},
                                "input_dimensions": {key: metadata.get(key) for key in ("max_events", "vocab_size", "n_static")}},
                               sort_keys=True)
        if family_signature is not None and signature != family_signature:
            raise ValueError("Run configurations/input dimensions differ; summarize each experimental family separately")
        family_signature = signature
        seed = config.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"Run must record an integer random seed: {directory}")
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed {seed}; do not count repeated fits as independent seeds")
        seen_seeds.add(seed)
        scores = load_aligned_predictions(directory / "val_predictions.csv", reference["patient_ids"], reference["target_codes"])
        runs.append({"run_dir": str(directory.resolve()), "seed": seed,
                     "probabilities": scores, "config": config})
    if expected_seeds is not None:
        expected = list(expected_seeds)
        if len(set(expected)) != len(expected) or set(expected) != seen_seeds:
            raise ValueError(f"Observed seeds {sorted(seen_seeds)} do not match all expected seeds {expected}")
    return sorted(runs, key=lambda run: run["seed"])


def _group_mask(values, n_patients, patient_ids):
    array = np.asarray(values)
    if array.dtype.kind == "b":
        if array.shape != (n_patients,):
            raise ValueError("Group boolean mask must have one entry per validation patient")
        return array
    if patient_ids is None:
        raise ValueError("Patient IDs are required when a group is specified by IDs")
    selected = list(values)
    if any(not isinstance(value, str) for value in selected) or len(set(selected)) != len(selected):
        raise ValueError("Group patient IDs must be unique strings")
    unknown = set(selected) - set(patient_ids)
    if unknown:
        raise ValueError(f"Group contains unknown validation patient IDs: {sorted(unknown)[:5]}")
    return np.isin(patient_ids, selected)


def evaluate_groups(
    y_true: Any, probabilities: Any, target_codes: Sequence[str], groups: Mapping[str, Any],
    patient_ids: Sequence[str] | None = None, target_descriptions=None,
) -> dict[str, Any]:
    """Describe short/long cohorts without silently dropping their rare targets.

    Raw scores between groups are not directly comparable when their prevalence
    and condition support differ. Compare models within the same group instead.
    """
    overall = evaluate_predictions(y_true, probabilities, target_codes, target_descriptions)
    labels = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    ids = list(patient_ids) if patient_ids is not None else None
    if ids is not None and (len(ids) != len(labels) or len(set(ids)) != len(ids)):
        raise ValueError("patient_ids must uniquely match validation rows")
    result = {}
    for name, values in groups.items():
        mask = _group_mask(values, len(labels), ids)
        count = int(mask.sum())
        result[str(name)] = {
            "n_patients": count,
            "positive_instances": int(labels[mask].sum()),
            "patients_with_any_positive": int((labels[mask].sum(axis=1) > 0).sum()),
            "n_positive_by_target": labels[mask].sum(axis=0).astype(int).tolist(),
            "metrics": evaluate_predictions(labels[mask], scores[mask], target_codes, target_descriptions) if count else None,
        }
    return {"overall": overall, "groups": result,
            "group_comparison_note": "Compare model differences within each fixed group; group prevalences/supports differ."}


def average_validation_predictions(
    run_dirs: Sequence[str | Path], prepared_dir: str | Path, expected_seeds=None,
) -> np.ndarray:
    """Equal mean of every supplied seed, aligned to the reference validation IDs."""
    reference = _validation_reference(prepared_dir)
    runs = _load_runs(run_dirs, reference, expected_seeds)
    return np.mean([run["probabilities"] for run in runs], axis=0)


def summarize_runs(
    run_dirs: Sequence[str | Path], prepared_dir: str | Path, groups: Mapping[str, Any] | None = None,
    *, family_name: str | None = None, expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe seed stability and an unweighted, all-seed ensemble.

    Call separately for each experimental family. ``expected_seeds`` can enforce
    an advance-declared seed list so a failed/unfavorable seed is not omitted.
    No test predictions or labels are opened by this function.
    """
    reference = _validation_reference(prepared_dir)
    runs = _load_runs(run_dirs, reference, expected_seeds)
    per_seed = []
    for run in runs:
        result = evaluate_groups(reference["labels"], run["probabilities"], reference["target_codes"],
                                 groups or {}, reference["patient_ids"], reference["target_descriptions"])
        per_seed.append({"run_dir": run["run_dir"], "seed": run["seed"],
                         "metrics": result["overall"], "groups": result["groups"]})
    aggregate = {}
    for key in ("mAP", "macro_auroc", "micro_ap", "brier"):
        values = [row["metrics"][key] for row in per_seed if row["metrics"][key] is not None]
        aggregate[key] = {"mean": float(np.mean(values)) if values else None,
                          "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                          "min": min(values) if values else None, "max": max(values) if values else None,
                          "n_defined": len(values)}
    ensemble_scores = np.mean([run["probabilities"] for run in runs], axis=0)
    ensemble = evaluate_groups(reference["labels"], ensemble_scores, reference["target_codes"], groups or {},
                               reference["patient_ids"], reference["target_descriptions"])
    return {"family": family_name, "split": "validation", "n_patients": len(reference["patient_ids"]),
            "target_codes": reference["target_codes"], "seeds": [run["seed"] for run in runs],
            "n_seeds": len(runs), "per_seed": per_seed, "across_seeds": aggregate,
            "equal_weight_ensemble": {"method": "uniform arithmetic probability mean over every listed seed",
                                      "metrics": ensemble["overall"], "groups": ensemble["groups"]},
            "uncertainty_note": UNCERTAINTY_NOTE}


def _score_orders(scores):
    orders = []
    for column in range(scores.shape[1]):
        order = np.argsort(-scores[:, column], kind="stable")
        ordered = scores[order, column]
        ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(order) - 1]
        orders.append((order, ends))
    return orders


def _weighted_macro_metrics(labels, orders, weights):
    """AP/AUC of patient multiplicities with exact score-tie grouping.

    This is equivalent to explicitly repeating each patient by its bootstrap
    count, but fixed-score ordering is calculated only once per model/class.
    """
    aps, aucs = [], []
    for column, (order, ends) in enumerate(orders):
        ordered_weights = weights[order]
        positive = ordered_weights * labels[order, column]
        cumulative_positive = np.cumsum(positive)[ends]
        cumulative_weight = np.cumsum(ordered_weights)[ends]
        positives = float(cumulative_positive[-1])
        negatives = float(cumulative_weight[-1] - positives)
        increments_positive = np.diff(np.r_[0.0, cumulative_positive])
        precision = np.divide(cumulative_positive, cumulative_weight,
                              out=np.zeros_like(cumulative_positive), where=cumulative_weight > 0)
        aps.append(float(np.sum(precision * increments_positive) / positives) if positives else 0.0)
        if positives and negatives:
            cumulative_negative = cumulative_weight - cumulative_positive
            negative_increments = np.diff(np.r_[0.0, cumulative_negative])
            previous_positive = np.r_[0.0, cumulative_positive[:-1]]
            area = np.sum(negative_increments * (cumulative_positive + previous_positive) / 2)
            aucs.append(float(area / (positives * negatives)))
    return {"mAP": float(np.mean(aps)), "macro_auroc": float(np.mean(aucs)) if aucs else None,
            "n_auroc_valid": len(aucs)}


def _bootstrap_interval(values, estimate, confidence):
    defined = np.asarray([value for value in values if value is not None], dtype=np.float64)
    alpha = (1 - confidence) / 2
    return {"estimate": estimate,
            "ci_lower": float(np.quantile(defined, alpha)) if len(defined) else None,
            "ci_upper": float(np.quantile(defined, 1 - alpha)) if len(defined) else None,
            "bootstrap_mean": float(defined.mean()) if len(defined) else None,
            "bootstrap_std": float(defined.std(ddof=1)) if len(defined) > 1 else None,
            "n_resamples_defined": len(defined),
            "fraction_delta_greater_than_zero": float((defined > 0).mean()) if len(defined) else None}


def paired_bootstrap(
    y_true: Any, probabilities_a: Any, probabilities_b: Any, target_codes: Sequence[str],
    n_bootstrap: int = 300, seed: int = 2026, confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired patient-percentile intervals for B minus A on the same cohort.

    All target APs, including zero-positive targets, contribute to every resample's
    mAP. AUC uses resample-specific classes having both positive/negative examples;
    this eligible set is identical for A and B. Undefined AUC resamples are counted
    explicitly and excluded only from the AUC interval, never the mAP interval.
    """
    codes = _validate_target_codes(target_codes)
    metric_a = evaluate_predictions(y_true, probabilities_a, codes)
    metric_b = evaluate_predictions(y_true, probabilities_b, codes)
    labels = np.asarray(y_true, dtype=np.float64)
    scores_a = validate_probability_matrix(probabilities_a, n_patients=len(labels), n_targets=len(codes))
    scores_b = validate_probability_matrix(probabilities_b, n_patients=len(labels), n_targets=len(codes))
    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    orders_a, orders_b = _score_orders(scores_a), _score_orders(scores_b)
    rng = np.random.default_rng(seed)
    map_deltas, auc_deltas, valid_class_counts = [], [], []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        weights = np.bincount(indices, minlength=len(labels)).astype(np.float64)
        a = _weighted_macro_metrics(labels, orders_a, weights)
        b = _weighted_macro_metrics(labels, orders_b, weights)
        map_deltas.append(b["mAP"] - a["mAP"])
        auc_deltas.append(b["macro_auroc"] - a["macro_auroc"] if a["macro_auroc"] is not None else None)
        valid_class_counts.append(a["n_auroc_valid"])
    point_a = {key: metric_a[key] for key in ("mAP", "macro_auroc", "n_auroc_valid")}
    point_b = {key: metric_b[key] for key in ("mAP", "macro_auroc", "n_auroc_valid")}
    auc_estimate = metric_b["macro_auroc"] - metric_a["macro_auroc"] if metric_a["macro_auroc"] is not None else None
    return {"difference_direction": "B minus A", "n_patients": len(labels), "n_targets": len(codes),
            "n_bootstrap": n_bootstrap, "seed": seed, "confidence_level": confidence,
            "resampling_unit": "patient; all conditions and both model predictions move together",
            "point_estimate": {"a": point_a, "b": point_b},
            "delta_mAP": _bootstrap_interval(map_deltas, metric_b["mAP"] - metric_a["mAP"], confidence),
            "delta_macro_auroc": _bootstrap_interval(auc_deltas, auc_estimate, confidence),
            "bootstrap_n_auroc_valid": {"min": min(valid_class_counts), "max": max(valid_class_counts),
                                         "mean": float(np.mean(valid_class_counts))},
            "metric_policy": "All target APs including no-positive AP=0; AUROC classes eligible separately per resample, identical eligibility for A/B.",
            "uncertainty_note": UNCERTAINTY_NOTE}
