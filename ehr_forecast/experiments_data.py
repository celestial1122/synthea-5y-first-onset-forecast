"""Train-fitted common encoders for controlled history-length experiments.

Unlike the first implementation, vocabulary frequency is measured over the
complete training histories, so changing the retained sequence length cannot
change the encoder. Patient-level statistics never use future test outcomes.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .data import (
    SPECIAL_TOKENS, STATIC_FEATURE_NAMES, TABLE_SPECS, _collect_events,
    _event_token, _fit_static, _load_anchors, _load_patients, _load_splits,
    _load_targets, _sha256, _write_json, first_diagnosis_labels, shift_years,
)


def _load_histories(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir).resolve()
    splits, ids = _load_splits(data_dir)
    codes, descriptions = _load_targets(data_dir)
    patients = _load_patients(data_dir, splits)
    anchors, end_dates = _load_anchors(data_dir, splits)
    events, first_dates, table_audit, train_values = _collect_events(data_dir, splits, anchors)
    # Match the original date/code/value ordering and duplicate rule exactly.
    histories = {
        patient: sorted(set(rows), key=lambda event: (
            event.day, event.token, -math.inf if event.numeric_value is None else event.numeric_value
        ))
        for patient, rows in events.items()
    }
    return dict(data_dir=data_dir, splits=splits, ids=ids, codes=codes,
                descriptions=descriptions, patients=patients, anchors=anchors,
                end_dates=end_dates, histories=histories, first_dates=first_dates,
                table_audit=table_audit, train_values=train_values)


def _quantiles(values) -> dict[str, float]:
    return dict(zip(["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"],
                    np.quantile(values, [0, .25, .5, .75, .9, .95, .99, 1]).astype(float).tolist()))


def _modalities(events) -> Counter:
    return Counter(event.token.split(":", 1)[0] for event in events)


def _clinical_codes(events, prefix):
    return {event.token for event in events if event.token.startswith(prefix + ":")}


def _write_diagnostics(state: dict, output_root: Path, lengths: tuple[int, ...]) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    by_split = {}
    patient_rows = []
    for split, patient_ids in state["ids"].items():
        counts, lookbacks, full_spans = [], [], []
        length_metrics = {
            length: dict(truncated=0, retained=0, dropped=0, retained_modalities=Counter(),
                         dropped_modalities=Counter(), retained_lookbacks=[], lost_lookbacks=[],
                         missing_cond=[], missing_med=[], patient_retention=[]) for length in lengths
        }
        for patient in patient_ids:
            events = state["histories"][patient]
            anchor = state["anchors"][patient].toordinal()
            count = len(events)
            full_lookback = (anchor - events[0].day) / 365.25 if events else 0.0
            full_span = (events[-1].day - events[0].day) / 365.25 if events else 0.0
            counts.append(count)
            lookbacks.append(full_lookback)
            full_spans.append(full_span)
            full_conditions, full_medications = [_clinical_codes(events, prefix) for prefix in ("COND", "MED")]
            row = dict(patient_id=patient, split=split, full_event_count=count,
                       history_group_256="long" if count > 255 else "short",
                       full_history_lookback_years=full_lookback,
                       full_observed_span_years=full_span,
                       full_unique_conditions=len(full_conditions), full_unique_medications=len(full_medications))
            for length in lengths:
                kept = events[-(length - 1):]
                removed = events[:max(0, count - (length - 1))]
                stats = length_metrics[length]
                stats["truncated"] += int(bool(removed))
                stats["retained"] += len(kept)
                stats["dropped"] += len(removed)
                stats["retained_modalities"].update(_modalities(kept))
                stats["dropped_modalities"].update(_modalities(removed))
                retained_lookback = (anchor - kept[0].day) / 365.25 if kept else 0.0
                lost_lookback = full_lookback - retained_lookback
                lost_cond = len(full_conditions - _clinical_codes(kept, "COND"))
                lost_med = len(full_medications - _clinical_codes(kept, "MED"))
                for name, value in [("retained_lookbacks", retained_lookback), ("lost_lookbacks", lost_lookback),
                                    ("missing_cond", lost_cond), ("missing_med", lost_med),
                                    ("patient_retention", len(kept) / max(count, 1))]:
                    stats[name].append(value)
                row.update({f"l{length}_retained_events": len(kept), f"l{length}_dropped_events": len(removed),
                            f"l{length}_retained_lookback_years": retained_lookback,
                            f"l{length}_lost_lookback_years": lost_lookback,
                            f"l{length}_lost_unique_conditions": lost_cond,
                            f"l{length}_lost_unique_medications": lost_med})
            patient_rows.append(row)
        formatted_lengths = {}
        for length, stats in length_metrics.items():
            formatted_lengths[str(length)] = {
                "truncated_patients": stats["truncated"], "truncated_fraction": stats["truncated"] / len(patient_ids),
                "retained_events": stats["retained"], "dropped_events": stats["dropped"],
                "aggregate_event_retention": stats["retained"] / max(1, sum(counts)),
                "retained_modalities": dict(stats["retained_modalities"]), "dropped_modalities": dict(stats["dropped_modalities"]),
                "retained_lookback_years": _quantiles(stats["retained_lookbacks"]),
                "lost_lookback_years": _quantiles(stats["lost_lookbacks"]),
                "lost_unique_conditions": _quantiles(stats["missing_cond"]),
                "lost_unique_medications": _quantiles(stats["missing_med"]),
                "patients_losing_unique_conditions": int(np.count_nonzero(stats["missing_cond"])),
                "patients_losing_unique_medications": int(np.count_nonzero(stats["missing_med"])),
                "patient_event_retention": _quantiles(stats["patient_retention"]),
            }
        long_counts = [row for row in patient_rows if row["split"] == split and row["history_group_256"] == "long"]
        by_split[split] = {
            "patients": len(patient_ids), "full_event_counts": _quantiles(counts),
            "full_history_lookback_years": _quantiles(lookbacks), "full_observed_span_years": _quantiles(full_spans),
            "long_history_patients_256": len(long_counts), "lengths": formatted_lengths,
            "among_long_256": {
                "patients": len(long_counts),
                "lost_lookback_years": _quantiles([row["l256_lost_lookback_years"] for row in long_counts]) if long_counts and 256 in lengths else None,
                "lost_unique_conditions": _quantiles([row["l256_lost_unique_conditions"] for row in long_counts]) if long_counts and 256 in lengths else None,
                "lost_unique_medications": _quantiles([row["l256_lost_unique_medications"] for row in long_counts]) if long_counts and 256 in lengths else None,
            },
        }
    diagnostics = {
        "schema_version": 1, "fit_or_label_selection": "No model or test outcome is used in these descriptive history statistics.",
        "event_rule": "pre-anchor calendar dates, deduplicated by day/code/value, original deterministic tie order",
        "lookback_definition": "anchor minus oldest event date; lost lookback is oldest retained date minus oldest full-history date",
        "sequence_lengths_include_bos": list(lengths), "splits": by_split,
    }
    _write_json(output_root / "history_diagnostics.json", diagnostics)
    with (output_root / "history_patient_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)
    return diagnostics


def analyze_histories(data_dir: Path, output_root: Path, lengths=(256, 512, 1024)) -> dict:
    """Quantify discarded chronology and clinical codes on the official splits."""
    lengths = tuple(sorted(set(int(value) for value in lengths)))
    if not lengths or min(lengths) < 2:
        raise ValueError("Lengths must include at least BOS and one event")
    return _write_diagnostics(_load_histories(data_dir), Path(output_root), lengths)


def prepare_length_ablation(
    data_dir: Path, output_root: Path, lengths=(256, 512, 1024),
    min_token_count: int = 3, n_value_bins: int = 8,
    *, _state: dict | None = None,
) -> dict:
    """Prepare different recent-history windows with exactly one fitted encoder.

    Encoder fitting sees all deduplicated training pre-anchor events, not only
    events retained by one length. The encoder JSON bytes are identical across
    lengths. Labels/prior status/static vectors and patient order are invariant.
    Existing caches are protected; use a fresh output root for a new recipe.
    """
    lengths = tuple(sorted(set(int(value) for value in lengths)))
    if not lengths or min(lengths) < 2 or min_token_count < 1 or n_value_bins < 2:
        raise ValueError("Need lengths >=2, min_token_count >=1 and n_value_bins >=2")
    output_root = Path(output_root).resolve()
    destinations = {length: output_root / f"prepared_l{length}" for length in lengths}
    if any((destination / "train.npz").exists() for destination in destinations.values()):
        raise FileExistsError("Length-ablation cache already exists; use a new output_root")
    state = _load_histories(data_dir) if _state is None else _state
    static, static_stats, static_audit = _fit_static(state["patients"], state["anchors"], state["ids"]["train"])
    quantiles = np.arange(1, n_value_bins) / n_value_bins
    bin_edges = {token: np.unique(np.quantile(values, quantiles)).tolist()
                 for token, values in sorted(state["train_values"].items())}
    token_counts = Counter(_event_token(event, bin_edges) for patient in state["ids"]["train"]
                           for event in state["histories"][patient])
    token_to_id = dict(SPECIAL_TOKENS)
    token_to_id.update({token: index + len(SPECIAL_TOKENS)
                       for index, token in enumerate(sorted(token for token, count in token_counts.items()
                                                            if count >= min_token_count))})
    vocabulary = {"token_to_id": token_to_id, "special_tokens": SPECIAL_TOKENS}
    preprocessing = {
        "schema_version": 2, "fit_split": "train", "vocabulary_scope": "all deduplicated pre-anchor training history",
        "numeric_bin_edges": bin_edges, "numeric_quantiles": quantiles.tolist(), "static": static_stats,
        "special_tokens": SPECIAL_TOKENS, "min_token_count": min_token_count, "n_value_bins": n_value_bins,
        "history_rule": "event calendar date < anchor calendar date",
        "label_rule": "anchor <= first-ever condition date <= last encounter date",
        "same_date_order": "clinical token string, then numeric value; no ordering inferred within a day",
        "truncation": "keep BOS and most recent max_events-1 events; max_events is an independent metadata field",
        "bos_time_days": 0.0,
        "event_token_format": "MODALITY:CODE; observations add UNIT and train-fitted BIN or categorical VALUE",
    }
    source_hashes = {}
    for name in ("patient_splits.csv", "target_conditions.csv", "test_anchors.csv"):
        source_hashes[name] = _sha256(state["data_dir"] / name)
    for folder in ("train_val", "test"):
        for name in ["patients.csv", *(f"{table}.csv" for table in TABLE_SPECS)]:
            path = state["data_dir"] / folder / name
            if path.exists():
                source_hashes[f"{folder}/{name}"] = _sha256(path)
    outputs = {}
    for length, destination in destinations.items():
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 2, "target_codes": state["codes"], "target_descriptions": state["descriptions"],
            "vocab_size": len(token_to_id), "n_static": len(STATIC_FEATURE_NAMES),
            "static_feature_names": STATIC_FEATURE_NAMES, "max_events": length,
            "split_counts": {split: len(values) for split, values in state["ids"].items()},
            "history_rule": preprocessing["history_rule"], "label_rule": preprocessing["label_rule"],
            "date_precision": "calendar date; ISO timestamp suffix discarded consistently",
            "min_token_count": min_token_count, "n_value_bins": n_value_bins,
            "modalities": list(TABLE_SPECS), "fit_split": "train",
            "vocabulary_scope": preprocessing["vocabulary_scope"],
            "static_inputs": ["BIRTHDATE (age at anchor)", "GENDER"],
            "ablation_control": "Common encoder across all tested sequence lengths; first-run encoder intentionally differs.",
        }
        audit = {
            "source_files_sha256": source_hashes, "tables": state["table_audit"], "static": static_audit,
            "splits": {}, "encoder_recipe": preprocessing["vocabulary_scope"],
            "conventions": ["Official split and complete-history incident labels are preserved.",
                            "No future event or validation/test patient fits any encoder statistic.",
                            "BOS time is fixed to zero and prior conditions use complete history.",
                            "Vocabulary/preprocessing JSONs are byte-identical across length variants."],
        }
        for split, patient_ids in state["ids"].items():
            count, targets = len(patient_ids), len(state["codes"])
            tokens = np.zeros((count, length), dtype=np.int64)
            time_days = np.zeros((count, length), dtype=np.float32)
            labels = np.zeros((count, targets), dtype=np.float32)
            prior = np.zeros((count, targets), dtype=np.bool_)
            unknown, retained_count, truncated, empty = 0, 0, 0, 0
            history_lengths, anchor_dates = [], []
            for index, patient in enumerate(patient_ids):
                full_history = state["histories"][patient]
                history = full_history[-(length - 1):]
                history_lengths.append(len(full_history))
                truncated += int(len(full_history) > length - 1)
                empty += int(not history)
                tokens[index, 0] = SPECIAL_TOKENS["[BOS]"]
                anchor = state["anchors"][patient]
                anchor_dates.append(anchor.isoformat())
                for position, event in enumerate(history, 1):
                    encoded = token_to_id.get(_event_token(event, bin_edges), SPECIAL_TOKENS["[UNK]"])
                    tokens[index, position] = encoded
                    time_days[index, position] = anchor.toordinal() - event.day
                    unknown += int(encoded == SPECIAL_TOKENS["[UNK]"])
                    retained_count += 1
                endpoint = state["end_dates"].get(patient, shift_years(anchor, 5))
                patient_labels, patient_prior = first_diagnosis_labels(
                    state["first_dates"][patient], anchor, endpoint, state["codes"])
                prior[index] = patient_prior
                if split != "test":
                    labels[index] = patient_labels
            arrays = {
                "patient_ids": np.asarray(patient_ids), "tokens": tokens, "time_days": time_days,
                "static": np.stack([static[patient] for patient in patient_ids]), "prior_conditions": prior,
                "anchor_dates": np.asarray(anchor_dates), "full_history_length": np.asarray(history_lengths, dtype=np.int32),
            }
            if split != "test":
                arrays["labels"] = labels
            np.savez_compressed(destination / f"{split}.npz", **arrays)
            audit["splits"][split] = {
                "patients": count, "empty_histories": empty, "truncated_histories": truncated,
                "retained_events": retained_count, "unknown_tokens": unknown,
                "unknown_token_fraction": unknown / max(retained_count, 1),
                "full_history_event_count_quantiles": _quantiles(history_lengths),
                "prior_target_counts": prior.sum(axis=0).astype(int).tolist(),
            }
            if split != "test":
                audit["splits"][split]["positive_label_counts"] = labels.sum(axis=0).astype(int).tolist()
        for filename, payload in [("metadata.json", metadata), ("preprocessing.json", preprocessing),
                                  ("vocabulary.json", vocabulary), ("audit.json", audit)]:
            _write_json(destination / filename, payload)
        outputs[str(length)] = {
            "path": str(destination), "vocab_size": len(token_to_id),
            "vocabulary_sha256": _sha256(destination / "vocabulary.json"),
            "preprocessing_sha256": _sha256(destination / "preprocessing.json"),
            "split_counts": metadata["split_counts"],
        }
    diagnostics = _write_diagnostics(state, output_root, lengths)
    manifest = {
        "schema_version": 1, "lengths": list(lengths), "outputs": outputs,
        "same_encoder_for_all_lengths": len({entry["vocabulary_sha256"] for entry in outputs.values()}) == 1
                                       and len({entry["preprocessing_sha256"] for entry in outputs.values()}) == 1,
        "initial_run_comparability": "Compare length variants to the new l256 control; old first-run vocabulary differs.",
    }
    _write_json(output_root / "length_ablation_manifest.json", manifest)
    return manifest


def prepare_summary_dataset(
    data_dir: Path, output_dir: Path, max_events: int = 256,
    min_token_count: int = 3, n_value_bins: int = 8,
    standardized_clip: float | None = None,
) -> dict:
    """Append full-history count/recent-count/recency features to safe static data.

    The three summary channels are per clinical token (including fitted numeric
    value bin), so older clinical findings survive a short compatibility token
    sequence. The feature vocabulary, all numeric bins, and scaling statistics
    see only pre-anchor training patients. No patient is dropped. An optional
    fixed symmetric clip affects only standardized summary channels, never the
    original demographic features. The default leaves the schema3 recipe bytes
    unchanged for existing checkpoint reproducibility.
    """
    if standardized_clip is not None:
        standardized_clip = float(standardized_clip)
        if not math.isfinite(standardized_clip) or standardized_clip <= 0:
            raise ValueError("standardized_clip must be finite and positive, or None")
    output_dir = Path(output_dir).resolve()
    if (output_dir / "train.npz").exists():
        raise FileExistsError("Summary cache already exists; use a new output_dir")
    state = _load_histories(data_dir)
    with TemporaryDirectory(prefix="ehr_summary_encoder_") as temporary:
        base_root = Path(temporary)
        manifest = prepare_length_ablation(
            data_dir, base_root, lengths=(max_events,), min_token_count=min_token_count,
            n_value_bins=n_value_bins, _state=state)
        base = Path(manifest["outputs"][str(max_events)]["path"])
        vocabulary = json.loads((base / "vocabulary.json").read_text(encoding="utf-8"))
        preprocessing = json.loads((base / "preprocessing.json").read_text(encoding="utf-8"))
        metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
        audit = json.loads((base / "audit.json").read_text(encoding="utf-8"))
        feature_tokens = [token for token, index in sorted(vocabulary["token_to_id"].items(), key=lambda item: item[1])
                          if index >= len(SPECIAL_TOKENS)]
        token_index = {token: index for index, token in enumerate(feature_tokens)}
        all_ids = [patient for patient_ids in state["ids"].values() for patient in patient_ids]
        features = np.zeros((len(all_ids), len(feature_tokens) * 3), dtype=np.float32)
        for row, patient in enumerate(all_ids):
            anchor = state["anchors"][patient]
            recent_start = shift_years(anchor, -2).toordinal()
            counts, recent_counts, nearest = Counter(), Counter(), {}
            for event in state["histories"][patient]:
                token = _event_token(event, preprocessing["numeric_bin_edges"])
                index = token_index.get(token)
                if index is None:
                    continue
                counts[index] += 1
                if event.day >= recent_start:
                    recent_counts[index] += 1
                days = anchor.toordinal() - event.day
                nearest[index] = min(nearest.get(index, days), days)
            for index, count in counts.items():
                features[row, index * 3] = math.log1p(count)
                features[row, index * 3 + 1] = math.log1p(recent_counts[index])
                features[row, index * 3 + 2] = math.exp(-nearest[index] / 3650.0)
        train_count = len(state["ids"]["train"])
        fit_features = features[:train_count].astype(np.float64)
        mean = fit_features.mean(axis=0)
        scale = fit_features.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = ((features.astype(np.float64) - mean) / scale).astype(np.float32)
        if standardized_clip is not None:
            np.clip(standardized, -standardized_clip, standardized_clip, out=standardized)
        summary_names = [f"summary_standardized::{channel}::{token}" for token in feature_tokens
                         for channel in ("log1p_all_count", "log1p_recent_2y_count", "recency_exp_10y")]
        metadata.update(
            schema_version=3, n_static=len(STATIC_FEATURE_NAMES) + len(summary_names),
            static_feature_names=[*STATIC_FEATURE_NAMES, *summary_names],
            static_inputs=[*metadata["static_inputs"], "All nine modalities strictly before anchor; full-history summary"],
            summary_feature_count=len(summary_names), full_history_summary=True,
        )
        preprocessing["schema_version"] = 3
        preprocessing["full_history_summary"] = {
            "feature_tokens": feature_tokens, "feature_names": summary_names,
            "mean": mean.tolist(), "scale": scale.tolist(), "fit_split": "train",
            "channels": ["log1p(all pre-anchor count)", "log1p(count in preceding two calendar years)",
                         "exp(-minimum days before anchor / 3650)"],
            "absent_token_values_before_scaling": [0.0, 0.0, 0.0],
            "vocabulary_scope": "same full-history train-only code/bin vocabulary as controlled length experiments",
            "recent_rule": "anchor minus two calendar years <= event date < anchor",
            "constant_feature_scale": 1.0,
        }
        if standardized_clip is not None:
            metadata["summary_standardized_clip"] = standardized_clip
            preprocessing["full_history_summary"]["standardized_clip"] = standardized_clip
        audit["summary_features"] = {
            "n_features": len(summary_names), "fit_patients": train_count, "fit_split": "train",
            "features_use_complete_pre_anchor_history": True, "tokens_remain_recent_window": max_events,
            "feature_names_file": "metadata.json:static_feature_names",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        offset = 0
        for split, patient_ids in state["ids"].items():
            with np.load(base / f"{split}.npz", allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            count = len(patient_ids)
            arrays["static"] = np.concatenate([arrays["static"], standardized[offset:offset + count]], axis=1)
            assert arrays["static"].shape == (count, metadata["n_static"])
            assert np.isfinite(arrays["static"]).all()
            np.savez_compressed(output_dir / f"{split}.npz", **arrays)
            offset += count
        shutil.copy2(base / "vocabulary.json", output_dir / "vocabulary.json")
        for name, payload in (("preprocessing.json", preprocessing), ("metadata.json", metadata), ("audit.json", audit)):
            _write_json(output_dir / name, payload)
    return metadata
