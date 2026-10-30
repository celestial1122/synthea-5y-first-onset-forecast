"""Explicit multi-label metrics, including behavior for rare conditions.

AUROC is undefined when a class has no positive or no negative examples. Such
classes are excluded only from macro AUROC, and their count is reported. The
primary mAP always averages *every* requested target: a target without a positive
example contributes zero. A positive-class-only AP is also reported separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_probability_matrix(
    probabilities: Any, *, n_patients: int, n_targets: int
) -> np.ndarray:
    """Return a finite [0, 1] matrix with the exact requested shape."""
    try:
        matrix = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Probabilities must be numeric.") from error
    expected_shape = (n_patients, n_targets)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected probability shape {expected_shape}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Probabilities contain NaN or infinity.")
    if ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1, inclusive.")
    return matrix


def _validate_target_codes(target_codes: Sequence[str]) -> list[str]:
    codes = list(target_codes)
    if not codes or any(not isinstance(code, str) or not code for code in codes):
        raise ValueError("Target codes must be nonempty strings.")
    if len(set(codes)) != len(codes):
        raise ValueError("Target codes must be unique.")
    return codes


def evaluate_predictions(
    y_true: Any,
    probabilities: Any,
    target_codes: Sequence[str],
    target_descriptions: Sequence[str] | Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate an entire split without silently omitting difficult targets.

    All return values can be saved as standards-compliant JSON (no NaN values).
    A validation split without any AUROC-eligible class gets ``None`` for macro
    AUROC. The caller should not replace that value with a fabricated score.
    """
    codes = _validate_target_codes(target_codes)
    labels = np.asarray(y_true)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] != len(codes):
        raise ValueError("Labels must have shape [nonzero patients, target codes].")
    try:
        labels = labels.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Labels must contain only 0 and 1.") from error
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only finite 0 and 1 values.")
    scores = validate_probability_matrix(
        probabilities, n_patients=labels.shape[0], n_targets=len(codes)
    )

    if target_descriptions is None:
        descriptions = [""] * len(codes)
    elif isinstance(target_descriptions, Mapping):
        descriptions = [str(target_descriptions.get(code, "")) for code in codes]
    else:
        descriptions = list(target_descriptions)
        if len(descriptions) != len(codes):
            raise ValueError("Descriptions must match the number of target codes.")

    rows: list[dict[str, Any]] = []
    valid_auroc: list[float] = []
    positive_class_aps: list[float] = []
    all_aps: list[float] = []
    for index, (code, description) in enumerate(zip(codes, descriptions)):
        truth = labels[:, index]
        n_positive = int(truth.sum())
        n_negative = int(len(truth) - n_positive)
        eligible = bool(n_positive and n_negative)
        auroc = float(roc_auc_score(truth, scores[:, index])) if eligible else None
        if auroc is not None:
            valid_auroc.append(auroc)
        ap = (
            float(average_precision_score(truth, scores[:, index]))
            if n_positive
            else 0.0
        )
        all_aps.append(ap)
        if n_positive:
            positive_class_aps.append(ap)
        rows.append(
            {
                "code": code,
                "description": str(description),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "prevalence": float(n_positive / len(truth)),
                "auroc": auroc,
                "auroc_defined": eligible,
                "average_precision": ap,
                "ap_has_positive_examples": bool(n_positive),
            }
        )

    flat_truth = labels.reshape(-1)
    flat_scores = scores.reshape(-1)
    n_positive_all = int(flat_truth.sum())
    micro_auroc = (
        float(roc_auc_score(flat_truth, flat_scores))
        if 0 < n_positive_all < len(flat_truth)
        else None
    )
    return {
        "n_patients": int(labels.shape[0]),
        "n_targets": len(codes),
        "macro_auroc": float(np.mean(valid_auroc)) if valid_auroc else None,
        "n_auroc_valid": len(valid_auroc),
        "mAP": float(np.mean(all_aps)),
        "map_positive_classes": (
            float(np.mean(positive_class_aps)) if positive_class_aps else None
        ),
        "n_ap_positive_classes": len(positive_class_aps),
        "micro_auroc": micro_auroc,
        "micro_ap": (
            float(average_precision_score(flat_truth, flat_scores))
            if n_positive_all
            else 0.0
        ),
        "brier": float(np.mean((scores - labels) ** 2)),
        "metric_policy": {
            "macro_auroc": "Mean of classes with both positive and negative labels.",
            "mAP": "Mean over ALL target classes; AP=0 for classes without positives.",
            "map_positive_classes": "Secondary metric, excludes classes without positives.",
        },
        "per_condition": rows,
    }
