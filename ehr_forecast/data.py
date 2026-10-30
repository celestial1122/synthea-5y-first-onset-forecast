"""Leakage-aware, train-fitted preparation of the supplied Synthea tables.

Date convention: all timestamps are reduced to their recorded calendar date.
History is strictly before the anchor; labels use [anchor, anchor + 5 years].
Only START/DATE and clinical code/value fields enter the model. STOP is read
solely to establish train/validation follow-up, never as a model feature.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np


SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[MASK]": 3}
TABLE_SPECS = {
    "conditions": ("START", "COND"),
    "medications": ("START", "MED"),
    "procedures": ("DATE", "PROC"),
    "observations": ("DATE", "OBS"),
    "encounters": ("START", "ENC"),
    "immunizations": ("DATE", "IMM"),
    "allergies": ("START", "ALLERGY"),
    "careplans": ("START", "CARE"),
    "devices": ("START", "DEVICE"),
}
STATIC_FEATURE_NAMES = [
    "age_at_anchor_standardized", "sex_female", "sex_male", "sex_unknown", "age_missing"
]


class _Event(NamedTuple):
    day: int
    token: str
    numeric_value: float | None


def _rows(path: Path, required: Iterable[str] = ()):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            yield line, row


def parse_date(value: str) -> date:
    """Use calendar-date precision consistently, including UTC ISO timestamps."""
    if not value or len(value.strip()) < 10:
        raise ValueError(f"Missing/invalid date: {value!r}")
    return date.fromisoformat(value.strip()[:10])


def shift_years(value: date, years: int) -> date:
    """Calendar-year arithmetic; Feb 29 maps to Feb 28 when necessary."""
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def first_diagnosis_labels(
    first_dates: dict[str, date], anchor: date, end: date, target_codes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(target_codes), dtype=np.float32)
    prior = np.zeros(len(target_codes), dtype=np.bool_)
    for index, code in enumerate(target_codes):
        first = first_dates.get(code)
        if first is not None:
            prior[index] = first < anchor
            labels[index] = float(anchor <= first <= end)
    return labels, prior


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _clean(value: str) -> str:
    # Encoding is lossless apart from whitespace normalization; no clinical text
    # dictionary, external data, or pretrained tokenizer is used.
    return " ".join(str(value or "").split())


def _event_token(event: _Event, bin_edges: dict[str, list[float]]) -> str:
    if event.numeric_value is None:
        return event.token
    edges = bin_edges.get(event.token)
    if edges is None:
        return event.token + "|NUM_UNSEEN"
    bin_index = int(np.searchsorted(edges, event.numeric_value, side="right"))
    return event.token + f"|BIN={bin_index}"


def _load_splits(data_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    mapping: dict[str, str] = {}
    ids = {name: [] for name in ("train", "val", "test")}
    for line, row in _rows(data_dir / "patient_splits.csv", ("Id", "split")):
        patient, split = row["Id"].strip(), row["split"].strip()
        if not patient or split not in ids:
            raise ValueError(f"patient_splits.csv:{line}: invalid patient/split")
        if patient in mapping:
            raise ValueError(f"Duplicate patient in patient_splits.csv: {patient}")
        mapping[patient] = split
        ids[split].append(patient)
    if any(not ids[name] for name in ids):
        raise ValueError("Each official split must contain patients")
    return mapping, ids


def _load_targets(data_dir: Path) -> tuple[list[str], list[str]]:
    codes, descriptions = [], []
    for _, row in _rows(data_dir / "target_conditions.csv", ("CODE", "DESCRIPTION")):
        code = row["CODE"].strip()
        if not code or code in codes:
            raise ValueError(f"Empty or duplicate target condition: {code!r}")
        codes.append(code)
        descriptions.append(row["DESCRIPTION"])
    if not codes:
        raise ValueError("No target conditions supplied")
    return codes, descriptions


def _load_patients(data_dir: Path, splits: dict[str, str]) -> dict[str, dict[str, str]]:
    patients = {}
    for folder, allowed in (("train_val", {"train", "val"}), ("test", {"test"})):
        path = data_dir / folder / "patients.csv"
        for line, row in _rows(path, ("Id", "BIRTHDATE", "GENDER")):
            patient = row["Id"].strip()
            if patient not in splits or splits[patient] not in allowed:
                raise ValueError(f"{path}:{line}: patient outside expected split: {patient}")
            if patient in patients:
                raise ValueError(f"Duplicate patient row: {patient}")
            patients[patient] = {"BIRTHDATE": row["BIRTHDATE"], "GENDER": row["GENDER"]}
    missing = set(splits) - set(patients)
    if missing:
        raise ValueError(f"patients.csv missing {len(missing)} official IDs: {sorted(missing)[:5]}")
    return patients


def _load_anchors(data_dir: Path, splits: dict[str, str]) -> tuple[dict[str, date], dict[str, date]]:
    end_dates: dict[str, date] = {}
    path = data_dir / "train_val" / "encounters.csv"
    for line, row in _rows(path, ("START", "STOP", "PATIENT")):
        patient = row["PATIENT"].strip()
        if splits.get(patient) not in {"train", "val"}:
            raise ValueError(f"{path}:{line}: invalid train/val patient {patient}")
        endpoint = parse_date(row["STOP"] or row["START"])
        if patient not in end_dates or endpoint > end_dates[patient]:
            end_dates[patient] = endpoint
    missing = {p for p, split in splits.items() if split != "test"} - set(end_dates)
    if missing:
        raise ValueError(f"Cannot calculate anchors for {len(missing)} train/val patients without encounters")
    anchors = {patient: shift_years(end, -5) for patient, end in end_dates.items()}
    test_ids = {p for p, split in splits.items() if split == "test"}
    seen = set()
    for line, row in _rows(data_dir / "test_anchors.csv", ("Id", "anchor_date")):
        patient = row["Id"].strip()
        if patient not in test_ids or patient in seen:
            raise ValueError(f"test_anchors.csv:{line}: unexpected or duplicate patient {patient}")
        seen.add(patient)
        anchors[patient] = parse_date(row["anchor_date"])
    if seen != test_ids:
        raise ValueError(f"test_anchors.csv missing {len(test_ids - seen)} official test patients")
    return anchors, end_dates


def _collect_events(
    data_dir: Path, splits: dict[str, str], anchors: dict[str, date]
) -> tuple[dict[str, list[_Event]], dict[str, dict[str, date]], dict[str, Any], dict[str, list[float]]]:
    events: dict[str, list[_Event]] = {patient: [] for patient in splits}
    first_dates: dict[str, dict[str, date]] = defaultdict(dict)
    table_audit = {}
    train_values: dict[str, list[float]] = defaultdict(list)
    for folder, allowed in (("train_val", {"train", "val"}), ("test", {"test"})):
        for table, (date_column, prefix) in TABLE_SPECS.items():
            path = data_dir / folder / f"{table}.csv"
            if not path.exists():
                # Smaller fixtures and explicitly documented optional tables may
                # omit these. The two core clinical tables are mandatory.
                if table in {"conditions", "encounters"}:
                    raise FileNotFoundError(path)
                table_audit[f"{folder}/{table}"] = {"available": False}
                continue
            counts: Counter = Counter()
            for line, row in _rows(path, ("PATIENT", date_column, "CODE")):
                patient = row["PATIENT"].strip()
                split = splits.get(patient)
                if split not in allowed:
                    raise ValueError(f"{path}:{line}: unexpected patient {patient}")
                event_date = parse_date(row[date_column])
                code = _clean(row["CODE"])
                if not code:
                    raise ValueError(f"{path}:{line}: empty clinical CODE")
                counts["rows"] += 1
                if table == "conditions":
                    previous = first_dates[patient].get(code)
                    if previous is None or event_date < previous:
                        first_dates[patient][code] = event_date
                if event_date >= anchors[patient]:
                    counts["excluded_on_or_after_anchor"] += 1
                    continue
                counts["pre_anchor_rows"] += 1
                token = f"{prefix}:{code}"
                numeric_value = None
                if table == "observations":
                    units = _clean(row.get("UNITS", "")) or "[NONE]"
                    token += f"|UNIT={units}"
                    raw_value = _clean(row.get("VALUE", ""))
                    if row.get("TYPE", "").lower() == "numeric":
                        try:
                            numeric_value = float(raw_value)
                            if not math.isfinite(numeric_value):
                                raise ValueError("non-finite")
                        except (TypeError, ValueError):
                            numeric_value = None
                            token += "|VALUE_MISSING"
                            counts["numeric_value_missing"] += 1
                        if numeric_value is not None and split == "train":
                            train_values[token].append(numeric_value)
                    else:
                        token += "|VALUE=" + (raw_value or "[MISSING]")
                events[patient].append(_Event(event_date.toordinal(), token, numeric_value))
            table_audit[f"{folder}/{table}"] = {"available": True, **dict(counts)}
    return events, first_dates, table_audit, train_values


def _fit_static(
    patients: dict[str, dict[str, str]], anchors: dict[str, date], train_ids: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    ages = {}
    for patient, row in patients.items():
        birth = row["BIRTHDATE"].strip()
        ages[patient] = (anchors[patient] - parse_date(birth)).days / 365.25 if birth else math.nan
    train_ages = np.asarray([ages[p] for p in train_ids], dtype=np.float64)
    finite = train_ages[np.isfinite(train_ages)]
    impute = float(np.median(finite)) if finite.size else 0.0
    imputed_train = np.where(np.isfinite(train_ages), train_ages, impute)
    mean, scale = float(imputed_train.mean()), float(imputed_train.std())
    scale = scale if scale > 1e-8 else 1.0
    result = {}
    for patient, row in patients.items():
        age = ages[patient]
        missing = not math.isfinite(age)
        sex = row["GENDER"].strip().upper()
        result[patient] = np.asarray([
            ((impute if missing else age) - mean) / scale,
            float(sex == "F"), float(sex == "M"), float(sex not in {"F", "M"}), float(missing),
        ], dtype=np.float32)
    stats = {"age_impute": impute, "age_mean": mean, "age_scale": scale}
    audit = {"age_missing": sum(not math.isfinite(v) for v in ages.values()),
             "anchors_before_birth": sum(math.isfinite(v) and v < 0 for v in ages.values())}
    return result, stats, audit


def prepare_dataset(
    data_dir: Path, output_dir: Path, max_events: int = 256,
    min_token_count: int = 3, n_value_bins: int = 8,
) -> dict[str, Any]:
    """Build reproducible arrays; every fitted statistic sees train history only.

    Patient ordering is the ordering of ``patient_splits.csv`` within each split.
    The saved preprocessing/vocabulary JSONs are the complete encoding recipe.
    Existing output files are replaced only after source validation completes.
    """
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if max_events < 2 or min_token_count < 1 or n_value_bins < 2:
        raise ValueError("Need max_events >= 2, min_token_count >= 1 and n_value_bins >= 2")
    splits, ids = _load_splits(data_dir)
    codes, descriptions = _load_targets(data_dir)
    patients = _load_patients(data_dir, splits)
    anchors, end_dates = _load_anchors(data_dir, splits)
    events, first_dates, table_audit, train_values = _collect_events(data_dir, splits, anchors)
    static, static_stats, static_audit = _fit_static(patients, anchors, ids["train"])
    quantiles = np.arange(1, n_value_bins) / n_value_bins
    bin_edges = {token: np.unique(np.quantile(values, quantiles)).tolist()
                 for token, values in sorted(train_values.items())}
    # Count only events actually presented to the model, rather than allowing
    # a discarded long history to determine min-frequency filtering.
    retained_events = {}
    deduplicated_counts = {}
    for patient, raw_events in events.items():
        unique = sorted(set(raw_events), key=lambda e: (e.day, e.token, -math.inf if e.numeric_value is None else e.numeric_value))
        deduplicated_counts[patient] = len(unique)
        retained_events[patient] = unique[-(max_events - 1):]
    token_counts = Counter(_event_token(event, bin_edges)
                           for patient in ids["train"] for event in retained_events[patient])
    token_to_id = dict(SPECIAL_TOKENS)
    token_to_id.update({token: index + len(SPECIAL_TOKENS)
                        for index, token in enumerate(sorted(t for t, count in token_counts.items()
                                                             if count >= min_token_count))})
    metadata = {
        "schema_version": 1, "target_codes": codes, "target_descriptions": descriptions,
        "vocab_size": len(token_to_id), "n_static": len(STATIC_FEATURE_NAMES),
        "static_feature_names": STATIC_FEATURE_NAMES, "max_events": max_events,
        "split_counts": {name: len(values) for name, values in ids.items()},
        "history_rule": "event calendar date < anchor calendar date",
        "label_rule": "anchor <= first-ever condition date <= last encounter date",
        "date_precision": "calendar date; ISO timestamp suffix discarded consistently",
        "min_token_count": min_token_count, "n_value_bins": n_value_bins,
        "modalities": list(TABLE_SPECS), "fit_split": "train",
        "static_inputs": ["BIRTHDATE (age at anchor)", "GENDER"],
    }
    audit: dict[str, Any] = {
        "source_files_sha256": {}, "tables": table_audit, "static": static_audit,
        "splits": {},
        "forbidden_feature_fields": [
            "STOP", "DEATHDATE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE",
            "DISPENSES", "TOTALCOST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE",
            "SSN", "DRIVERS", "PASSPORT", "FIRST", "LAST", "REASONCODE", "REASONDESCRIPTION",
        ],
        "conventions": [
            "Official patient split is preserved; no patient is dropped.",
            "Train/validation anchor = calendar latest(STOP if nonblank else START) minus 5 calendar years.",
            "Test anchors are read directly from test_anchors.csv.",
            "History is strictly before anchor; anchor-date diagnoses belong to the future label window.",
            "Labels use first-ever diagnosis across complete train/validation condition history.",
            "Only train pre-anchor history fits numeric quantiles, static scaling, and vocabulary.",
            "Same-date identical code/value events are deduplicated; codes break same-date ties deterministically.",
            "Latest max_events-1 events are kept after BOS; static age remains available for empty histories.",
            "BOS time_days is fixed to zero; it never copies the first clinical event time.",
            "No imaging/genomics, external datasets, or pretrained weights are used.",
        ],
        "unresolved_specification": [
            "README does not specify anchor-day inclusion; this implementation uses a documented disjoint date convention.",
            "Test outcomes are hidden; test AUROC/mAP must be supplied by the evaluator, not estimated locally.",
        ],
    }
    for filename in ("patient_splits.csv", "target_conditions.csv", "test_anchors.csv"):
        audit["source_files_sha256"][filename] = _sha256(data_dir / filename)
    for folder in ("train_val", "test"):
        for filename in ["patients.csv", *(f"{table}.csv" for table in TABLE_SPECS)]:
            path = data_dir / folder / filename
            if path.exists():
                audit["source_files_sha256"][f"{folder}/{filename}"] = _sha256(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, patient_ids in ids.items():
        count = len(patient_ids)
        tokens = np.zeros((count, max_events), dtype=np.int64)
        time_days = np.zeros((count, max_events), dtype=np.float32)
        prior_conditions = np.zeros((count, len(codes)), dtype=np.bool_)
        labels = np.zeros((count, len(codes)), dtype=np.float32)
        lengths, unknown, total_tokens, empty, truncated = [], 0, 0, 0, 0
        anchor_values = []
        for index, patient in enumerate(patient_ids):
            history = retained_events[patient]
            tokens[index, 0] = SPECIAL_TOKENS["[BOS]"]
            if not history:
                empty += 1
            if deduplicated_counts[patient] > max_events - 1:
                truncated += 1
            lengths.append(len(history))
            for position, event in enumerate(history, 1):
                token = _event_token(event, bin_edges)
                encoded = token_to_id.get(token, SPECIAL_TOKENS["[UNK]"])
                tokens[index, position] = encoded
                time_days[index, position] = anchors[patient].toordinal() - event.day
                unknown += int(encoded == SPECIAL_TOKENS["[UNK]"])
                total_tokens += 1
            anchor_values.append(anchors[patient].isoformat())
            # Test first_dates are only pre-anchor records, so only prior status
            # is used there. Hidden test outcomes are never fabricated.
            endpoint = end_dates.get(patient, shift_years(anchors[patient], 5))
            patient_labels, patient_prior = first_diagnosis_labels(first_dates[patient], anchors[patient], endpoint, codes)
            prior_conditions[index] = patient_prior
            if name != "test":
                labels[index] = patient_labels
        arrays = {
            "patient_ids": np.asarray(patient_ids), "tokens": tokens, "time_days": time_days,
            "static": np.stack([static[p] for p in patient_ids]), "prior_conditions": prior_conditions,
            "anchor_dates": np.asarray(anchor_values),
        }
        if name != "test":
            arrays["labels"] = labels
        np.savez_compressed(output_dir / f"{name}.npz", **arrays)
        audit["splits"][name] = {
            "patients": count, "empty_histories": empty, "truncated_histories": truncated,
            "retained_events": int(total_tokens), "unknown_tokens": int(unknown),
            "unknown_token_fraction": unknown / max(total_tokens, 1),
            "history_event_count_quantiles": dict(zip(["min", "p25", "p50", "p75", "max"],
                                                        np.quantile(lengths, [0, .25, .5, .75, 1]).tolist())),
            "prior_target_counts": prior_conditions.sum(axis=0).astype(int).tolist(),
        }
        if name != "test":
            audit["splits"][name]["positive_label_counts"] = labels.sum(axis=0).astype(int).tolist()
            audit["splits"][name]["no_positive_target_patients"] = int((labels.sum(axis=1) == 0).sum())
    preprocessing = {
        "schema_version": 1, "fit_split": "train", "numeric_bin_edges": bin_edges,
        "numeric_quantiles": quantiles.tolist(), "static": static_stats,
        "special_tokens": SPECIAL_TOKENS, "max_events": max_events,
        "history_rule": metadata["history_rule"], "label_rule": metadata["label_rule"],
        "event_token_format": "MODALITY:CODE; observations also include UNIT and train-fitted BIN or categorical VALUE",
        "same_date_order": "clinical token string, then numeric value; no ordering inferred within a day",
        "truncation": "keep BOS and most recent max_events-1 distinct events",
        "bos_time_days": 0.0,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "vocabulary.json", {"token_to_id": token_to_id, "special_tokens": SPECIAL_TOKENS})
    _write_json(output_dir / "preprocessing.json", preprocessing)
    _write_json(output_dir / "audit.json", audit)
    return metadata
