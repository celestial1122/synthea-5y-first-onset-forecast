"""Strict submission serialization and independent CSV validation."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import _validate_target_codes, validate_probability_matrix


def _patient_ids(values: Iterable[str], *, label: str) -> list[str]:
    ids = list(values)
    if not ids:
        raise ValueError(f"{label} cannot be empty.")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in ids
    ):
        raise ValueError(f"{label} must contain nonempty, unpadded string IDs.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate patient IDs.")
    return ids


def _check_coverage(ids: list[str], expected_patient_ids: Iterable[str] | None) -> None:
    if expected_patient_ids is None:
        return
    expected = _patient_ids(expected_patient_ids, label="Expected patient IDs")
    missing = sorted(set(expected) - set(ids))
    extra = sorted(set(ids) - set(expected))
    if missing or extra:
        raise ValueError(
            "Patient coverage mismatch: "
            f"missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}."
        )


def write_predictions(
    path: str | Path,
    patient_ids: Iterable[str],
    probabilities: Any,
    target_codes: Sequence[str],
    expected_patient_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write an atomic CSV only after validating the entire proposed submission."""
    destination = Path(path)
    codes = _validate_target_codes(target_codes)
    if "patient_id" in codes:
        raise ValueError("A condition code cannot use the patient_id column name.")
    ids = _patient_ids(patient_ids, label="Prediction patient IDs")
    expected = list(expected_patient_ids) if expected_patient_ids is not None else None
    _check_coverage(ids, expected)
    scores = validate_probability_matrix(
        probabilities, n_patients=len(ids), n_targets=len(codes)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.writer(stream)
            writer.writerow(["patient_id", *codes])
            for patient_id, row in zip(ids, scores):
                writer.writerow([patient_id, *(format(float(value), ".10g") for value in row)])
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return validate_predictions(destination, codes, expected if expected is not None else ids)


def validate_predictions(
    path: str | Path,
    target_codes: Sequence[str],
    expected_patient_ids: Iterable[str],
) -> dict[str, Any]:
    """Check literal column order, duplicates, coverage, shape, and probabilities.

    Uses the CSV reader directly so duplicate column names cannot be silently
    renamed by a dataframe library. Row order is arbitrary; coverage must be exact.
    """
    source = Path(path)
    codes = _validate_target_codes(target_codes)
    if "patient_id" in codes:
        raise ValueError("A condition code cannot use the patient_id column name.")
    ids: list[str] = []
    rows: list[list[float]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Prediction CSV is empty.") from error
        if len(header) != len(set(header)):
            raise ValueError("Prediction CSV has duplicate column headers.")
        expected_header = ["patient_id", *codes]
        if header != expected_header:
            raise ValueError(
                "CSV columns must be exactly patient_id followed by target CODEs "
                "in target_conditions.csv order."
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise ValueError(
                    f"CSV line {line_number} has {len(row)} fields; "
                    f"expected {len(expected_header)}."
                )
            ids.append(row[0])
            try:
                rows.append([float(value) for value in row[1:]])
            except ValueError as error:
                raise ValueError(f"Non-numeric probability at CSV line {line_number}.") from error
    ids = _patient_ids(ids, label="Prediction CSV patient IDs")
    _check_coverage(ids, expected_patient_ids)
    scores = validate_probability_matrix(
        np.asarray(rows), n_patients=len(ids), n_targets=len(codes)
    )
    return {
        "path": str(source.resolve()),
        "n_patients": len(ids),
        "n_targets": len(codes),
        "probability_min": float(scores.min()),
        "probability_max": float(scores.max()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "valid": True,
    }
