"""Create a self-contained local model bundle without contacting any service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


REQUIRED_ARTIFACTS = (
    "model_config.json",
    "metadata.json",
    "vocabulary.json",
    "preprocessing.json",
    "validation_metrics.json",
    "summary.json",
    "environment.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(value: Any) -> str:
    return "undefined for this validation split" if value is None else f"{float(value):.6f}"


def _model_card(metadata: dict, model_config: dict, metrics: dict, summary: dict, training: dict) -> str:
    model_type = model_config.get("model_type", "transformer")
    max_events = int(metadata["max_events"])
    counts = metadata.get("split_counts", {})
    modalities = ", ".join(metadata.get("modalities", [])) or "structured clinical events"
    if model_type in {"compressed_transformer", "hierarchical_transformer"}:
        bag_description = "sum divided by sqrt(event count)" if model_config.get("bag_normalization") == "sqrt" else "mean EmbeddingBag"
        architecture = (
            f"An order-invariant per-date event encoder uses {bag_description} plus log event count, "
            "absolute date position and days before the known anchor. "
        )
        if model_type == "hierarchical_transformer":
            architecture += (f"A Hi-BEHRT-inspired hierarchical Transformer applies {model_config.get('local_layers', 1)} "
                             f"local layer in windows of {model_config.get('window_size', 32)} dates "
                             f"with stride {model_config.get('window_stride', 32)}, then "
                             f"{model_config.get('global_layers', 2)} global layers over pooled windows. "
                             "It is a small-data adaptation, not a reproduction of the original paper or its pretraining.")
        else:
            architecture += f"A flat {model_config.get('n_layers', 3)}-layer Transformer attends over event/date units."
        architecture += (" All attention is within pre-anchor history. Masked mean and last-unit state "
                         "are fused with static demographics in a shared small classification head. "
                         "This is one fitted model; predictions do not average random seeds.")
    elif model_type in {"transformer", "pooled_transformer"}:
        architecture = (
            f"A decoder-only causal Transformer with {model_config.get('n_layers', 3)} layers, "
            f"{model_config.get('n_heads', 4)} attention heads, and hidden width "
            f"{model_config.get('d_model', 128)}. Classification uses the last valid event "
            "state concatenated with static features. Event position embeddings and "
            "scaled log/linear days before the known anchor encode order and time. "
            "Optional next-event pretraining shares the same causal backbone; the "
            "checkpoint's training_config.json records whether it was enabled."
        )
        if model_type == "pooled_transformer":
            architecture += " The classifier uses masked mean plus last-event pooling and nonlinear static-feature fusion."
    elif model_type in {"summary_mlp", "summary_linear"}:
        architecture = (
            "A PyTorch model of full-history summary features. It uses a small LayerNorm/GELU/dropout MLP "
            "with nonlinear demographic interactions." if model_type == "summary_mlp" else
            "A regularized PyTorch linear logistic classifier of the same full-history summary features."
        )
        architecture += " Token sequences are carried for interface compatibility but are not used by this classifier."
    else:
        architecture = (
            "An order-independent PyTorch bag-of-events baseline. Event embeddings "
            "are summed and divided by the square root of the event count, then "
            "combined with static features in a linear disease classification head. "
            "Event frequencies are retained; event order and timestamps are not "
            "used by this baseline."
        )
    wandb_mode = training.get("wandb_mode", "offline")
    static_note = (
        "Static inputs include demographic features plus train-standardized log1p lifetime event counts, "
        "log1p counts in the last two calendar years, and exponential recency for every clinical token/bin. "
        "These summaries use the complete pre-anchor history, including events outside the sequence window."
        if metadata.get("schema_version") == 3 else
        "Static inputs are age at the anchor and sex/missingness indicators."
    )
    if metadata.get("input_layout") == "event_units":
        sequence_note = (f"Inputs contain at most {max_events} right-padded event/date units, each an unordered event bag. "
                         "The raw control uses one event per unit; date controls keep all events inside each retained date. "
                         "Only an empty history gets a BOS sentinel. Detailed limits and the representation are in metadata.json.")
        capacity_limit = (f"A maximum of {max_events} event/date units can still discard the oldest units in extreme histories. "
                          "Within-date event order is intentionally not modeled; there is no within-retained-date truncation.")
    else:
        sequence_note = f"Sequences contain BOS plus the most recent {max_events - 1} pre-anchor events and are right-padded to at most {max_events} tokens."
        capacity_limit = ("Full-history summaries retain all pre-anchor dates but lose detailed event ordering and use quantized observations."
                          if metadata.get("schema_version") == 3 else
                          f"A maximum of {max_events} tokens can discard distant but relevant history.")
    if metadata.get("input_layout") == "event_units":
        visit_name = {"raw_recent511": "prepared_raw", "date_recent511": "prepared_date_recent", "date_full": "prepared_date_full"}[metadata["representation"]]
        max_unit_events = metadata.get("max_unit_events", 128)
        # The raw control's serialized bag width is one, but the common visit
        # recipe also reconstructs date controls and needs their fixed width.
        width = max(128, max_unit_events)
        raw_limit = metadata.get("raw_event_limit") or 511
        preparation_command = (f"python -m ehr_forecast prepare-visits --data-dir research_intern_coding_test "
                               f"--output-root artifacts/hierarchical --max-units {max_events} "
                               f"--max-unit-events {width} --raw-event-limit {raw_limit}")
        prepared_path = f"artifacts/hierarchical/{visit_name}"
    elif metadata.get("schema_version", 1) >= 2:
        mode = "summary" if metadata.get("schema_version") == 3 else "length"
        preparation_command = f"python -m ehr_forecast prepare-experiment --mode {mode} --data-dir research_intern_coding_test --output-root artifacts/optimization"
        prepared_path = "artifacts/optimization/prepared_summary" if mode == "summary" else f"artifacts/optimization/prepared_l{max_events}"
        if mode == "summary" and metadata.get("summary_standardized_clip") is not None:
            clip = metadata["summary_standardized_clip"]
            preparation_command += f" --summary-clip {clip:g}"
            prepared_path += f"_clip{clip:g}"
    else:
        preparation_command = f"python -m ehr_forecast prepare --data-dir research_intern_coding_test --output-dir artifacts/prepared --max-events {max_events} --min-token-count {metadata.get('min_token_count', 3)} --n-value-bins {metadata.get('n_value_bins', 8)}"
        prepared_path = "artifacts/prepared"
    return f"""---
library_name: pytorch
tags:
  - pytorch
  - synthetic-ehr
  - multi-label-classification
  - electronic-health-records
---

# Patient timeline forecasting ({model_type})

This directory is a **local export prepared for a future Hugging Face upload**.
Creating it does not upload the model or complete any public submission.

## Intended research use

Predict first-ever diagnoses of {len(metadata['target_codes'])} target conditions
within five years after a patient-specific anchor, using only earlier synthetic
electronic health records. This is an educational research model trained on the
provided Synthea dataset; it has not been validated for clinical decision-making
or real patient populations.

All weights were initialized from scratch. No external datasets, pretrained
model weights, pretrained tokenizers, imaging, or genomic inputs were used.
The fixed official splits contain {counts.get('train', 'unknown')} training,
{counts.get('val', 'unknown')} validation, and {counts.get('test', 'unknown')} test
patients. Validation is used only for monitoring and model selection.

## Model and representation

{architecture}

Structured modalities: {modalities}. Codes are modality-prefixed. Numeric
observation values are quantile-binned separately by observation code and unit;
all bin edges, the vocabulary, and age normalization are fitted only on training
patients. {static_note}
Names, identifiers, future death dates, lifetime cost totals, and future STOP
values are not model features.

{sequence_note} Special IDs are PAD=0, UNK=1,
BOS=2, and reserved MASK=3. The decoder objective does not use MASK corruption.
Exact token definitions, fitted transforms, and input dimensions are preserved
in vocabulary.json, preprocessing.json, and metadata.json.

Date precision is the recorded calendar date. For training/validation, each
encounter contributes STOP or START if STOP is blank; the latest such date minus
five calendar years is the anchor. February 29 maps to February 28 when needed.
For test patients, supplied test_anchors.csv dates are used directly.
History requires event date strictly less than the anchor. A positive label
requires the patient's earliest-ever diagnosis for that code to fall on or after
the anchor and on or before the last encounter date. Thus an anchor-day diagnosis
is a label, not an input. This date-boundary choice is explicit because the task
description does not fully specify anchor-day handling.

Conditions known before the anchor cannot become a first-ever diagnosis:
predict_probabilities explicitly sets those condition probabilities to **0**.
The corresponding targets are masked out of supervised training loss. Models
output independent sigmoid probabilities, so disease probabilities need not sum
to one. The target column order is metadata.json's target_codes.

## Measured validation results

- Validation patients: {metrics.get('n_patients', counts.get('val', 'unknown'))}.
- Macro AUROC: {_score(metrics.get('macro_auroc'))}, averaged over
  {metrics.get('n_auroc_valid', 'unknown')} targets with both label classes.
- mAP: {_score(metrics.get('mAP'))}, averaged over **all** target conditions.
- Micro average precision: {_score(metrics.get('micro_ap'))}.
- Brier score: {_score(metrics.get('brier'))}.
- Selected fine-tuning epoch: {summary.get('best_epoch', 'unknown')}.

For a target with no validation positives, AP contributes zero to mAP and AUROC
is undefined. Such targets are not silently removed from mAP. Per-condition
supports, AUROC availability, and AP are in validation_metrics.json.
**Test ground truth is hidden; test AUROC/mAP are unknown and are not reported.**
Validation scores select this checkpoint and are not an unbiased replacement for
the hidden test evaluation.

## Limits

Training is limited to a small fully synthetic cohort, with rare conditions and
potentially unstable per-condition estimates. {capacity_limit} The selected checkpoint uses a single
validation split; probability calibration and real-world generalization have not
been established. Historical conditions are tracked from the full available
pre-anchor diagnosis table even when their event tokens fall outside the recent
sequence window. Optional imaging and genomic modalities are not used.

## Reproduce inference from the official release

Use Python 3.11-3.13 and a suitable local PyTorch installation. Place the supplied
dataset (patient_splits.csv, target_conditions.csv, test_anchors.csv, train_val/,
and test/) under ./research_intern_coding_test. Run from this bundle directory:

```sh
python -m pip install -e .
{preparation_command}
python -m ehr_forecast predict --checkpoint model.pt --prepared-dir {prepared_path} --output predictions.csv --device auto --split test
python -m ehr_forecast validate-submission --predictions predictions.csv --data-dir research_intern_coding_test
```

The current preparation command recreates the encoder using the **official
training data**, then transforms validation/test patients. Therefore reproducing
raw-table inference requires that training release as well as the test tables.
Bundled vocabulary/preprocessing files document the frozen encoder; this export
does not yet provide a separate raw-table-only transform command for novel
patients. Prediction checks vocabulary/preprocessing SHA256 hashes and target
order against model.pt and rejects mismatches. Use the exact provided release,
not an alternative train split. No raw or prepared patient data is bundled.

model.pt is the selected best checkpoint, not the last training epoch. The
training configuration is included as training_config.json when present; configs/
contains reusable starting configurations. environment.json records the actual
training software and hardware. Reproduction across different hardware or
PyTorch releases can have small floating-point differences.

## Experiment tracking and publication status

This run used wandb mode **{wandb_mode}**. Offline wandb files remain in the local
training run directory and are intentionally excluded from this bundle. If the
run was offline, authenticate and run `wandb sync <local-offline-run-directory>`
before adding the resulting public or reviewer-accessible run URL to the final
report. No online wandb URL is invented here. Public GitHub/Hugging Face URLs and
the report's final identity fields still require the owner's publication step.

## Data attribution

Training data is derived from the **Coherent Synthetic Data Set**, generated by
Synthea and released under Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/ .

Copyright 2022, The MITRE Corporation. MITRE Public Release Case Number: 21-1917.
This model was trained on processed structured tables from the supplied task
release. Attribution is preserved here; the source dataset is not redistributed
inside this export. See DATA_ATTRIBUTION.md for the same provenance separately.

## Bundle integrity

manifest.json records SHA256 hashes and byte sizes for all other packaged files.
It excludes its own hash to avoid self-reference. Patient-level predictions,
raw/prepared arrays, and wandb payloads are deliberately absent.
"""


def export_model(run_dir: Path, output_dir: Path, source_root: Path | None = None) -> dict:
    """Copy a completed run into a new, local-only Hugging Face-ready folder.

    Existing nonempty destinations are never overwritten. Only explicitly
    allowlisted training artifacts, Python package sources, and configuration
    files are included; this function performs no network or upload operation.
    """
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    source_root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[1]
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Export destination must be absent or empty: {output_dir}")
    for name in ("best.pt", *REQUIRED_ARTIFACTS):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Completed run is missing required export artifact: {run_dir / name}")
    if not (source_root / "ehr_forecast").is_dir() or not (source_root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"Source root must contain ehr_forecast/ and pyproject.toml: {source_root}")

    metadata = _read_json(run_dir / "metadata.json")
    model_config = _read_json(run_dir / "model_config.json")
    metrics = _read_json(run_dir / "validation_metrics.json")
    summary = _read_json(run_dir / "summary.json")
    training = _read_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else {}
    checkpoint_hash = _sha256(run_dir / "best.pt")
    expected_hash = summary.get("checkpoint_sha256")
    if expected_hash is not None and expected_hash != checkpoint_hash:
        raise ValueError("best.pt does not match the completed run summary SHA256")
    if metrics.get("n_targets") != len(metadata["target_codes"]):
        raise ValueError("Validation metrics and model target count disagree")
    if summary.get("test_labels_available") is True or summary.get("test_metrics") is not None:
        raise ValueError("Export expects hidden test labels and no claimed test metrics")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "best.pt", output_dir / "model.pt")
    for name in REQUIRED_ARTIFACTS:
        shutil.copy2(run_dir / name, output_dir / name)
    if (run_dir / "group_validation_metrics.json").is_file():
        shutil.copy2(run_dir / "group_validation_metrics.json", output_dir / "group_validation_metrics.json")
    if (run_dir / "config.json").is_file():
        shutil.copy2(run_dir / "config.json", output_dir / "training_config.json")
    shutil.copy2(source_root / "pyproject.toml", output_dir / "pyproject.toml")
    if (source_root / "artifacts" / "environment-lock.txt").is_file():
        shutil.copy2(source_root / "artifacts" / "environment-lock.txt", output_dir / "requirements-lock.txt")
    for source in sorted((source_root / "ehr_forecast").rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        destination = output_dir / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    config_root = source_root / "configs"
    if config_root.is_dir():
        for source in sorted(config_root.rglob("*")):
            if source.is_file() and source.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}:
                destination = output_dir / source.relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    (output_dir / "README.md").write_text(
        _model_card(metadata, model_config, metrics, summary, training), encoding="utf-8"
    )
    (output_dir / "DATA_ATTRIBUTION.md").write_text(
        "# Training data attribution\n\n"
        "Derived from the Coherent Synthetic Data Set, generated by Synthea.\n\n"
        "Copyright 2022, The MITRE Corporation. MITRE Public Release Case Number: 21-1917.\n\n"
        "Source data license: Creative Commons Attribution 4.0 International "
        "(https://creativecommons.org/licenses/by/4.0/).\n\n"
        "The task release was processed into chronological clinical-code/value tokens "
        "and train-fitted numerical bins for synthetic future-condition forecasting. "
        "No raw or prepared patient records are redistributed in this bundle.\n",
        encoding="utf-8",
    )
    files = {
        path.relative_to(output_dir).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "local_export_only": True,
        "uploaded": False,
        "checkpoint_sha256": checkpoint_hash,
        "files": files,
        "manifest_policy": "Hashes cover all other bundle files; manifest.json excludes itself.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "files": len(files) + 1,
        "checkpoint_sha256": checkpoint_hash,
        "validation_mAP": metrics.get("mAP"),
        "test_metrics": None,
        "uploaded": False,
    }
