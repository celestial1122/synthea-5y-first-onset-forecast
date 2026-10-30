"""Paired, single-model statistics for a fixed two-factor experiment.

The estimands are linear contrasts of metrics computed *separately* for each
fitted seed. Patient resamples are shared across all four cells and every seed.
This module never averages probabilities or automatically attributes causation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .experiment_analysis import (
    UNCERTAINTY_NOTE, _bootstrap_interval, _group_mask, _score_orders,
    _weighted_macro_metrics,
)
from .metrics import _validate_target_codes, evaluate_predictions, validate_probability_matrix


CELLS = ("C0R0", "C1R0", "C0R1", "C1R1")
COEFFICIENTS = {
    "C_main": dict(zip(CELLS, (-.5, .5, -.5, .5))),
    "R_main": dict(zip(CELLS, (-.5, -.5, .5, .5))),
    "interaction": dict(zip(CELLS, (1., -1., -1., 1.))),
}
FORMULAS = {
    "C_main": "0.5 * [(C1R0 - C0R0) + (C1R1 - C0R1)]",
    "R_main": "0.5 * [(C0R1 - C0R0) + (C1R1 - C1R0)]",
    "interaction": "C1R1 - C1R0 - C0R1 + C0R0",
}
BCE_EPSILON = 1e-7
CAUTION = (
    "This is a controlled factorial description, conditional on the supplied fitted models, "
    "fixed seeds and validation cohort. Causal attribution additionally requires the declared "
    "experimental controls and implementation checks; the function cannot verify them from predictions. "
    "Main effects average over the other factor's two levels; inspect interaction before "
    "treating an effect as uniform. Multiple contrasts, metrics and subgroups are reported "
    "without simultaneous/multiplicity correction. Positive bootstrap fractions are descriptive, "
    "not p-values. Validation checkpoint/model selection is not undone by bootstrap."
)


def _summary(values):
    defined = [float(value) for value in values if value is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "sample_std": float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
        "min": min(defined) if defined else None,
        "max": max(defined) if defined else None,
        "n_defined": len(defined),
    }


def _wins(values, lower_is_better=False):
    # For an interaction, these are signs of metric synergy, not predictor wins.
    direction = -1 if lower_is_better else 1
    utility = [direction * float(value) for value in values if value is not None]
    return {
        "wins": sum(value > 1e-12 for value in utility),
        "ties": sum(abs(value) <= 1e-12 for value in utility),
        "losses": sum(value < -1e-12 for value in utility),
        "n_defined": len(utility),
        "improvement_direction": "negative" if lower_is_better else "positive",
        "tie_tolerance": 1e-12,
    }


def _contrast(values, coefficients):
    if any(values[cell] is None for cell, coefficient in coefficients.items() if coefficient):
        return None
    return float(sum(coefficients[cell] * values[cell] for cell in CELLS if coefficients[cell]))


def _validate_ids(values, n_patients):
    try:
        ids = list(values)
    except TypeError as error:
        raise ValueError("patient_ids must uniquely match every label row") from error
    if (len(ids) != n_patients or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("patient_ids must uniquely match every label row")
    return ids


def _validate_predictions(predictions, labels, codes, ids, expected_seeds, require_metadata):
    if not isinstance(predictions, Mapping) or set(predictions) != set(CELLS):
        raise ValueError(f"Exactly the four factorial cells are required: {CELLS}")
    scores, seed_set, metadata_count = {}, None, 0
    for cell in CELLS:
        runs = predictions[cell]
        if not isinstance(runs, Mapping) or not runs:
            raise ValueError(f"Cell {cell} needs a nonempty seed-to-prediction mapping")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in runs):
            raise ValueError("Prediction seeds must be nonnegative integers")
        if seed_set is not None and set(runs) != seed_set:
            raise ValueError("All four cells must contain the same complete seed set")
        seed_set = set(runs)
        scores[cell] = {}
        for model_seed, value in runs.items():
            if isinstance(value, Mapping):
                if not {"probabilities", "patient_ids", "target_codes"}.issubset(value):
                    raise ValueError("Prediction metadata needs probabilities, patient_ids and target_codes")
                own_ids = _validate_ids(value["patient_ids"], len(labels))
                if ids is None:
                    ids = own_ids
                if own_ids != ids:
                    raise ValueError("Prediction patient order differs from the reference; align rows explicitly")
                if list(value["target_codes"]) != codes:
                    raise ValueError("Prediction target CODE order differs from the reference")
                value = value["probabilities"]
                metadata_count += 1
            elif require_metadata:
                raise ValueError("Every prediction requires explicit patient/CODE order metadata")
            scores[cell][model_seed] = validate_probability_matrix(
                value, n_patients=len(labels), n_targets=len(codes))
    seeds = sorted(seed_set)
    if expected_seeds is not None:
        expected = list(expected_seeds)
        if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in expected)
                or len(set(expected)) != len(expected) or set(expected) != seed_set):
            raise ValueError("Observed seeds do not match the unique expected seeds")
    return scores, seeds, ids, metadata_count


def _validate_prior(prior_conditions, labels):
    if prior_conditions is None:
        return None
    try:
        prior = np.asarray(prior_conditions, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("prior_conditions must be a binary patient/target matrix") from error
    if prior.shape != labels.shape or not np.isfinite(prior).all() or not np.isin(prior, [0, 1]).all():
        raise ValueError("prior_conditions must be a binary matrix matching labels")
    prior = prior.astype(bool)
    if labels[prior].any():
        raise ValueError("First-ever outcome labels cannot overlap prior conditions")
    return prior


def _cohort_analysis(labels, scores, codes, seeds, prior, n_bootstrap, seed, confidence):
    metrics = ("mAP", "macro_auroc", "brier", "at_risk_bce") if prior is not None else ("mAP", "macro_auroc", "brier")
    eligible = ~prior if prior is not None else None
    risk_counts = eligible.sum(axis=1).astype(float) if eligible is not None else None
    points, orders, brier_rows, bce_rows = {}, {}, {}, {}
    cells = {}
    for cell in CELLS:
        points[cell], orders[cell], brier_rows[cell], bce_rows[cell] = {}, {}, {}, {}
        per_seed = []
        for model_seed in seeds:
            probabilities = scores[cell][model_seed]
            result = evaluate_predictions(labels, probabilities, codes)
            if eligible is not None:
                clipped = np.clip(probabilities, BCE_EPSILON, 1 - BCE_EPSILON)
                losses = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
                numerators = (losses * eligible).sum(axis=1)
                result["at_risk_bce"] = float(numerators.sum() / risk_counts.sum()) if risk_counts.sum() else None
                bce_rows[cell][model_seed] = numerators
            points[cell][model_seed] = result
            orders[cell][model_seed] = _score_orders(probabilities)
            brier_rows[cell][model_seed] = ((probabilities - labels) ** 2).mean(axis=1)
            per_seed.append({"seed": model_seed, "metrics": result})
        cells[cell] = {
            "per_seed": per_seed,
            "across_seeds": {metric: _summary([points[cell][value][metric] for value in seeds]) for metric in metrics},
        }
    contrasts = {}
    for name, coefficients in COEFFICIENTS.items():
        per_seed = [{"seed": value, **{
            metric: _contrast({cell: points[cell][value][metric] for cell in CELLS}, coefficients)
            for metric in metrics
        }} for value in seeds]
        summaries = {metric: _summary([row[metric] for row in per_seed]) for metric in metrics}
        contribution_rows = []
        for index, code in enumerate(codes):
            effects = [{"seed": value, "average_precision": _contrast({
                cell: points[cell][value]["per_condition"][index]["average_precision"] for cell in CELLS
            }, coefficients)} for value in seeds]
            effect = float(np.mean([row["average_precision"] for row in effects]))
            contribution_rows.append({"code": code, "n_positive": int(labels[:, index].sum()),
                                      "n_negative": int(len(labels) - labels[:, index].sum()),
                                      "mean_average_precision_contrast": effect,
                                      "contribution_to_mAP_contrast": effect / len(codes), "per_seed": effects})
        contrasts[name] = {
            "formula": FORMULAS[name], "coefficients": coefficients,
            "per_seed": per_seed, "between_seed": summaries,
            "mean_single_model_contrast": {metric: summaries[metric]["mean"] for metric in metrics},
            "wins": {metric: _wins([row[metric] for row in per_seed], metric in {"brier", "at_risk_bce"}) for metric in metrics},
            "wins_interpretation": "Signs of the contrast under the metric's improvement direction; an interaction sign is not a standalone predictor win.",
            "per_condition_ap_contributions": contribution_rows,
        }
    draws = {name: {metric: [] for metric in metrics} for name in COEFFICIENTS}
    valid_counts = []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        weights = np.bincount(indices, minlength=len(labels)).astype(float)
        weighted = {}
        for cell in CELLS:
            weighted[cell] = {}
            for model_seed in seeds:
                result = _weighted_macro_metrics(labels, orders[cell][model_seed], weights)
                result["brier"] = float(np.dot(weights, brier_rows[cell][model_seed]) / weights.sum())
                if eligible is not None:
                    denominator = np.dot(weights, risk_counts)
                    result["at_risk_bce"] = float(np.dot(weights, bce_rows[cell][model_seed]) / denominator) if denominator else None
                weighted[cell][model_seed] = result
        valid_counts.append(weighted[CELLS[0]][seeds[0]]["n_auroc_valid"])
        for name, coefficients in COEFFICIENTS.items():
            for metric in metrics:
                values = [_contrast({cell: weighted[cell][value][metric] for cell in CELLS}, coefficients) for value in seeds]
                draws[name][metric].append(float(np.mean(values)) if all(value is not None for value in values) else None)
    for name in COEFFICIENTS:
        contrasts[name]["intervals"] = {
            metric: _bootstrap_interval(draws[name][metric], contrasts[name]["between_seed"][metric]["mean"], confidence)
            for metric in metrics
        }
    return {
        "estimand": "Factorial contrasts of metrics computed separately for every fitted seed, then averaged across fixed paired seeds",
        "n_patients": len(labels), "n_targets": len(codes), "seeds": seeds,
        "n_positive_by_target": labels.sum(axis=0).astype(int).tolist(),
        "n_eligible_target_cells": int(eligible.sum()) if eligible is not None else None,
        "cells": cells, "contrasts": contrasts,
        "n_bootstrap": n_bootstrap, "bootstrap_seed": seed, "confidence_level": confidence,
        "bootstrap_n_auroc_valid": {"min": min(valid_counts), "max": max(valid_counts), "mean": float(np.mean(valid_counts))},
        "uses_probability_ensemble": False,
        "metric_policy": "mAP averages all requested targets (AP=0 without positives); AUROC excludes label-undefined classes identically across all cells/seeds on each draw; Brier averages all target cells.",
        "bce_policy": f"At-risk BCE excludes known prior conditions and clips probabilities to [{BCE_EPSILON}, {1-BCE_EPSILON}]" if eligible is not None else "At-risk BCE is not computed without an explicit prior-condition mask",
        "resampling_unit": "One shared patient multiplicity vector across all targets, four cells and fixed seeds; seeds are not resampled",
        "uncertainty_note": UNCERTAINTY_NOTE + " " + CAUTION,
    }


def factorial_analysis(
    y_true: Any,
    predictions: Mapping[str, Mapping[int, Any]],
    target_codes: Sequence[str],
    groups: Mapping[str, Any] | None = None,
    patient_ids: Sequence[str] | None = None,
    n_bootstrap: int = 500,
    seed: int = 2026,
    expected_seeds: Sequence[int] | None = (17, 42, 73),
    *,
    confidence: float = .95,
    prior_conditions: Any | None = None,
    require_alignment_metadata: bool = False,
) -> dict[str, Any]:
    """Analyze a complete C-by-R experiment without selecting a seed or model.

    Plain matrices must already share patient/target order; permutation cannot
    be detected from their numeric values. For strict order checks provide each
    seed as ``{probabilities, patient_ids, target_codes}`` and optionally require
    metadata for every seed via ``require_alignment_metadata=True``. Such
    payloads must match the reference exactly; strict mode also requires an
    explicit ``patient_ids`` reference for the label rows. Merely agreeing with
    another prediction's IDs cannot verify label-row order. This function never
    reorders arrays.
    Group definitions are fixed IDs or boolean masks, never inferred from scores.
    """
    codes = _validate_target_codes(target_codes)
    labels = np.asarray(y_true)
    evaluate_predictions(labels, np.zeros_like(labels, dtype=float), codes)
    labels = np.asarray(labels, dtype=float)
    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer >= 2")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if groups is not None and not isinstance(groups, Mapping):
        raise ValueError("groups must map names to fixed masks or patient IDs")
    if not isinstance(require_alignment_metadata, bool):
        raise ValueError("require_alignment_metadata must be boolean")
    if require_alignment_metadata and patient_ids is None:
        raise ValueError("Strict alignment requires reference patient_ids for the label rows")
    ids = _validate_ids(patient_ids, len(labels)) if patient_ids is not None else None
    scores, seeds, ids, metadata_count = _validate_predictions(
        predictions, labels, codes, ids, expected_seeds, require_alignment_metadata)
    prior = _validate_prior(prior_conditions, labels)
    controls = dict(codes=codes, seeds=seeds, n_bootstrap=n_bootstrap, seed=seed, confidence=confidence)
    overall = _cohort_analysis(labels, scores, prior=prior, **controls)
    group_results, counts = {}, {}
    for name, values in (groups or {}).items():
        name = str(name)
        if name in counts:
            raise ValueError("Group names must remain unique when converted to strings")
        mask = _group_mask(values, len(labels), ids)
        counts[name] = int(mask.sum())
        selected_scores = {cell: {value: matrix[mask] for value, matrix in runs.items()} for cell, runs in scores.items()}
        group_results[name] = _cohort_analysis(labels[mask], selected_scores, prior=prior[mask] if prior is not None else None,
                                             **controls) if mask.any() else None
    total = len(CELLS) * len(seeds)
    return {
        "overall": overall, "groups": group_results, "group_patient_counts": counts,
        "target_codes": codes, "seeds": seeds, "uses_probability_ensemble": False,
        "alignment": {"metadata_checked_predictions": metadata_count, "total_predictions": total,
                      "all_prediction_metadata_present": metadata_count == total,
                      "label_order_reference_provided": patient_ids is not None,
                      "strict_all_predictions": metadata_count == total and patient_ids is not None,
                      "policy": "Payload row IDs and CODE order are checked exactly; bare arrays require caller-verified alignment and cannot expose silent permutations"},
        "selection": None,
        "inference_scope": CAUTION,
    }
