"""Date-bag compression with an existing, frozen train-only clinical encoder.

Within-date clinical events are an unordered bag. No event inside a retained
date is silently discarded, and no new encoder statistic is fitted here.
"""

from __future__ import annotations

from collections import Counter
import copy
import csv
from itertools import groupby
import json
import math
from pathlib import Path
import shutil

import numpy as np

from .data import (
    SPECIAL_TOKENS, STATIC_FEATURE_NAMES, _event_token, _sha256, _write_json,
    first_diagnosis_labels, parse_date, shift_years,
)
from .experiments_data import _load_histories, _quantiles


REPRESENTATIONS = {
    "raw_recent511": "prepared_raw",
    "date_recent511": "prepared_date_recent",
    "date_full": "prepared_date_full",
}


def _date_groups(history):
    return [(day, list(events)) for day, events in groupby(history, key=lambda event: event.day)]


def analyze_date_units(data_dir: Path, output_root: Path, unit_limits=(128, 256, 512), *, _state=None) -> dict:
    """Describe compression coverage without using outcomes to choose a model."""
    state = _load_histories(data_dir) if _state is None else _state
    result = {}
    patient_rows = []
    for split, ids in state["ids"].items():
        grouped = {patient: _date_groups(state["histories"][patient]) for patient in ids}
        full_events = sum(len(state["histories"][patient]) for patient in ids)
        long_ids = [patient for patient in ids if len(state["histories"][patient]) > 255]
        long_events = sum(len(state["histories"][patient]) for patient in long_ids)
        unit_counts = [len(grouped[patient]) for patient in ids]
        daily_counts = [len(events) for patient in ids for _, events in grouped[patient]]
        lengths = {}
        for limit in unit_limits:
            retained = sum(sum(len(events) for _, events in grouped[patient][-limit:]) for patient in ids)
            long_retained = sum(sum(len(events) for _, events in grouped[patient][-limit:]) for patient in long_ids)
            lengths[str(limit)] = {
                "patients_truncated": sum(len(grouped[patient]) > limit for patient in ids),
                "event_retention": retained / max(full_events, 1),
                "long_patients_truncated": sum(len(grouped[patient]) > limit for patient in long_ids),
                "long_event_retention": long_retained / max(long_events, 1),
            }
        result[split] = {
            "patients": len(ids), "long_patients": len(long_ids), "date_units": _quantiles(unit_counts),
            "daily_event_count": _quantiles(daily_counts) if daily_counts else {}, "lengths": lengths,
        }
        for patient in ids:
            patient_rows.append({"patient_id": patient, "split": split,
                                 "full_event_count": len(state["histories"][patient]),
                                 "full_date_count": len(grouped[patient]),
                                 "history_group_256": "long" if patient in long_ids else "short"})
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "date_unit_diagnostics.json", result)
    with (output_root / "date_patient_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)
    return result


def _frozen_static(state, recipe):
    stats = recipe["static"]
    required = {"age_impute", "age_mean", "age_scale"}
    if not required.issubset(stats) or stats["age_scale"] <= 0:
        raise ValueError("Frozen encoder is missing valid training age statistics")
    result = {}
    for patient, row in state["patients"].items():
        birth = row["BIRTHDATE"].strip()
        age = (state["anchors"][patient] - parse_date(birth)).days / 365.25 if birth else math.nan
        missing = not math.isfinite(age)
        sex = row["GENDER"].strip().upper()
        result[patient] = np.asarray([
            ((stats["age_impute"] if missing else age) - stats["age_mean"]) / stats["age_scale"],
            float(sex == "F"), float(sex == "M"), float(sex not in {"F", "M"}), float(missing),
        ], dtype=np.float32)
    return result


def _units(history, representation, raw_event_limit):
    if representation == "raw_recent511":
        return [(event.day, [event]) for event in history[-raw_event_limit:]]
    if representation == "date_recent511":
        return _date_groups(history[-raw_event_limit:])
    return _date_groups(history)


def prepare_visit_datasets(
    data_dir: Path, encoder_dir: Path, output_root: Path,
    max_units: int = 512, max_unit_events: int = 128, raw_event_limit: int = 511,
) -> dict:
    """Prepare raw-event and date-compressed controls under one frozen encoder.

    ``raw_recent511`` and ``date_recent511`` encode exactly the same recent raw
    events. ``date_full`` groups the complete history before keeping max_units
    recent dates. Tokens use int16 storage and must be converted to torch.long.
    The names remain stable; non-default limits are explicit in every recipe.
    """
    data_dir, encoder_dir, output_root = [Path(path).resolve() for path in (data_dir, encoder_dir, output_root)]
    if not isinstance(max_units, int) or isinstance(max_units, bool) or max_units < 1:
        raise ValueError("max_units must be a positive integer")
    if not isinstance(max_unit_events, int) or isinstance(max_unit_events, bool) or max_unit_events < 1:
        raise ValueError("max_unit_events must be a positive integer")
    if not isinstance(raw_event_limit, int) or isinstance(raw_event_limit, bool) or not 1 <= raw_event_limit <= max_units:
        raise ValueError("raw_event_limit must be positive and <= max_units")
    destinations = {key: output_root / value for key, value in REPRESENTATIONS.items()}
    if any(path.exists() and any(path.iterdir()) for path in destinations.values()):
        raise FileExistsError("Visit cache output already exists; use a fresh output_root")
    source_metadata = json.loads((encoder_dir / "metadata.json").read_text(encoding="utf-8"))
    source_recipe = json.loads((encoder_dir / "preprocessing.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((encoder_dir / "vocabulary.json").read_text(encoding="utf-8"))
    if source_metadata.get("n_static") != len(STATIC_FEATURE_NAMES) or source_recipe.get("fit_split") != "train":
        raise ValueError("Use the frozen full-training clinical encoder with five demographic features")
    token_to_id = vocabulary["token_to_id"]
    if max(token_to_id.values()) >= np.iinfo(np.int16).max + 1 or min(token_to_id.values()) < 0:
        raise ValueError("Vocabulary IDs must fit nonnegative int16 storage")
    if any(token_to_id.get(token) != index for token, index in SPECIAL_TOKENS.items()):
        raise ValueError("Frozen special token IDs differ from the model contract")
    state = _load_histories(data_dir)
    if source_metadata["target_codes"] != state["codes"]:
        raise ValueError("Frozen target order differs from the supplied release")
    static = _frozen_static(state, source_recipe)
    # Freeze before encoding; no frequency, bin or scaler fit occurs below.
    bin_edges = source_recipe["numeric_bin_edges"]
    diagnostics = analyze_date_units(data_dir, output_root, _state=state)
    base_hashes = {name: _sha256(encoder_dir / name) for name in ("metadata.json", "vocabulary.json", "preprocessing.json")}
    outputs = {}
    for representation, destination in destinations.items():
        event_width = 1 if representation == "raw_recent511" else max_unit_events
        metadata = copy.deepcopy(source_metadata)
        metadata.update(
            schema_version=4, input_layout="event_units", representation=representation,
            max_events=max_units, max_unit_events=event_width, storage_token_dtype="int16",
            raw_event_limit=raw_event_limit if representation != "date_full" else None,
            unit_definition="one event" if representation == "raw_recent511" else "one pre-anchor calendar date",
            long_history_definition="more than 255 complete pre-anchor deduplicated events",
        )
        recipe = copy.deepcopy(source_recipe)
        recipe.update(schema_version=4, input_layout="event_units", representation=representation,
                      raw_event_limit=metadata["raw_event_limit"], max_unit_events=event_width,
                      frozen_base_encoder_sha256=base_hashes,
                      truncation="Keep recent units up to metadata.max_events; never truncate events inside a retained date",
                      unit_order="ascending calendar day; within-date token order is serialization only and has no clinical order",
                      empty_history="one BOS token in one unit at time zero; remaining slots are PAD")
        audit = {
            "frozen_base_encoder_sha256": base_hashes, "fit_split": "train (already-fitted source encoder)",
            "encoder_refitted": False, "tables": state["table_audit"], "splits": {},
            "long_history_definition": metadata["long_history_definition"],
            "conventions": [
                "Official patient IDs and split ordering are preserved.",
                "All real events have date < anchor; future rows are labels only.",
                "Full-history first diagnosis and prior-condition masks survive all unit truncation.",
                "Same-date events are unordered; distinct raw values that share a numerical bin retain their event multiplicity.",
                "No within-unit truncation; exceeding max_unit_events raises an error.",
            ],
        }
        destination.mkdir(parents=True, exist_ok=True)
        for split, ids in state["ids"].items():
            count = len(ids)
            tokens = np.zeros((count, max_units, event_width), dtype=np.int16)
            times = np.zeros((count, max_units), dtype=np.float32)
            prior = np.zeros((count, len(state["codes"])), dtype=np.bool_)
            labels = np.zeros((count, len(state["codes"])), dtype=np.float32)
            full_lengths, full_dates, retained_counts, retained_units = [], [], [], []
            totals = Counter()
            long_totals = Counter()
            anchors = []
            for index, patient in enumerate(ids):
                history = state["histories"][patient]
                units = _units(history, representation, raw_event_limit)
                retained = units[-max_units:]
                anchor = state["anchors"][patient]
                anchors.append(anchor.isoformat())
                full_lengths.append(len(history))
                full_dates.append(len(_date_groups(history)))
                retained_units.append(len(retained))
                retained_count = sum(len(events) for _, events in retained)
                retained_counts.append(retained_count)
                patient_stats = {"full_events": len(history), "retained_events": retained_count,
                                 "truncated_patients": int(retained_count < len(history)),
                                 "unit_truncated_patients": int(len(units) > max_units),
                                 "patients": 1}
                totals.update(patient_stats)
                if len(history) > 255:
                    long_totals.update(patient_stats)
                if not retained:
                    tokens[index, 0, 0] = SPECIAL_TOKENS["[BOS]"]
                    totals["empty_histories"] += 1
                for position, (day, events) in enumerate(retained):
                    if len(events) > event_width:
                        raise ValueError(f"Retained unit has {len(events)} events, exceeding max_unit_events={event_width}; patient={patient}")
                    encoded = [token_to_id.get(_event_token(event, bin_edges), SPECIAL_TOKENS["[UNK]"]) for event in events]
                    tokens[index, position, :len(encoded)] = encoded
                    times[index, position] = anchor.toordinal() - day
                    totals["unknown_events"] += sum(value == SPECIAL_TOKENS["[UNK]"] for value in encoded)
                endpoint = state["end_dates"].get(patient, shift_years(anchor, 5))
                patient_labels, patient_prior = first_diagnosis_labels(state["first_dates"][patient], anchor, endpoint, state["codes"])
                prior[index] = patient_prior
                if split != "test":
                    labels[index] = patient_labels
            arrays = {
                "patient_ids": np.asarray(ids), "tokens": tokens, "time_days": times,
                "static": np.stack([static[patient] for patient in ids]), "prior_conditions": prior,
                "anchor_dates": np.asarray(anchors), "full_history_length": np.asarray(full_lengths, dtype=np.int32),
                "full_unit_length": np.asarray(full_dates, dtype=np.int32),
                "retained_event_count": np.asarray(retained_counts, dtype=np.int32),
                "retained_unit_count": np.asarray(retained_units, dtype=np.int32),
            }
            if split != "test":
                arrays["labels"] = labels
            np.savez_compressed(destination / f"{split}.npz", **arrays)
            audit["splits"][split] = {
                **dict(totals), "aggregate_event_retention": totals["retained_events"] / max(totals["full_events"], 1),
                "full_history_event_count_quantiles": _quantiles(full_lengths),
                "full_date_count_quantiles": _quantiles(full_dates),
                "retained_unit_count_quantiles": _quantiles(retained_units),
                "prior_target_counts": prior.sum(axis=0).astype(int).tolist(),
                "long_history": {**dict(long_totals), "aggregate_event_retention": long_totals["retained_events"] / max(long_totals["full_events"], 1)},
            }
            if split != "test":
                audit["splits"][split]["positive_label_counts"] = labels.sum(axis=0).astype(int).tolist()
            # Release each dense int16 split before allocating the next one.
            del tokens, times, prior, labels, arrays
        _write_json(destination / "metadata.json", metadata)
        _write_json(destination / "preprocessing.json", recipe)
        _write_json(destination / "audit.json", audit)
        shutil.copy2(encoder_dir / "vocabulary.json", destination / "vocabulary.json")
        outputs[representation] = {
            "path": str(destination), "max_events": max_units, "max_unit_events": event_width,
            "vocabulary_sha256": _sha256(destination / "vocabulary.json"),
            "preprocessing_sha256": _sha256(destination / "preprocessing.json"),
            "split_retention": {split: item["aggregate_event_retention"] for split, item in audit["splits"].items()},
        }
    manifest = {"schema_version": 1, "outputs": outputs, "frozen_base_encoder_sha256": base_hashes,
                "max_units": max_units, "max_unit_events": max_unit_events, "raw_event_limit": raw_event_limit,
                "same_vocabulary": len({item["vocabulary_sha256"] for item in outputs.values()}) == 1,
                "date_diagnostics": diagnostics}
    _write_json(output_root / "visit_preparation_manifest.json", manifest)
    return manifest
