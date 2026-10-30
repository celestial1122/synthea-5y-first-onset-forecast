"""Equal-weight checkpoint inference and portable local ensemble exports.

Each member retains its own encoder and input dimensions.  Test predictions are
used only to produce the declared uniform average; no weights are fitted and no
test outcomes are read.  Exporting never contacts a service.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .metrics import _validate_target_codes, evaluate_predictions
from .submission import _patient_ids, validate_predictions, write_predictions


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _normalize_members(members):
    if not isinstance(members, (list, tuple)) or not members:
        raise ValueError("Provide a nonempty list of checkpoint/prepared_dir members")
    result, seen_checkpoints, reference_codes = [], set(), None
    for member in members:
        if not isinstance(member, dict) or not {"checkpoint", "prepared_dir"}.issubset(member):
            raise ValueError("Each member requires explicit checkpoint and prepared_dir")
        checkpoint = Path(member["checkpoint"]).resolve()
        prepared = Path(member["prepared_dir"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint in seen_checkpoints:
            raise ValueError("Duplicate checkpoint member would give it extra ensemble weight")
        seen_checkpoints.add(checkpoint)
        config_path = checkpoint.parent / "config.json"
        if not config_path.is_file():
            config_path = checkpoint.parent / "training_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Member needs its complete local training configuration: {checkpoint.parent}")
        config = _json(config_path)
        if not isinstance(config, dict) or not config or "seed" not in config or "model_type" not in config:
            raise ValueError(f"Incomplete member training configuration: {config_path}")
        metadata = _json(prepared / "metadata.json")
        codes = _validate_target_codes(metadata["target_codes"])
        if reference_codes is not None and set(codes) != set(reference_codes):
            raise ValueError("Ensemble members must predict the same target CODE set")
        if reference_codes is None:
            reference_codes = codes
        result.append({"checkpoint": checkpoint, "prepared_dir": prepared,
                       "config_path": config_path, "config": config, "metadata": metadata,
                       "target_codes": codes, "checkpoint_sha256": _sha256(checkpoint),
                       "encoder_sha256": {name: _sha256(prepared / name) for name in ("vocabulary.json", "preprocessing.json")}})
    return result, reference_codes


def _read_ids(prepared, split):
    with np.load(prepared / f"{split}.npz", allow_pickle=False) as arrays:
        return _patient_ids(arrays["patient_ids"].astype(str).tolist(), label="Prepared patient IDs")


def _read_aligned_csv(path, own_codes, expected_ids, output_codes):
    validate_predictions(path, own_codes, expected_ids)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {row["patient_id"]: [float(row[code]) for code in output_codes] for row in reader}
    return np.asarray([rows[patient] for patient in expected_ids], dtype=np.float64)


def _predict_member(checkpoint, prepared_dir, output_path, device, split):
    from .train import predict
    return predict(checkpoint, prepared_dir, output_path, device=device, split=split)


def predict_ensemble(members: list[dict], output_path: Path, device="auto", split="test") -> dict:
    """Predict every member with its own encoder, then average probabilities.

    Output IDs and column order follow the first member's prepared dataset;
    subsequent row/column orders can differ but coverage and CODE sets must be
    identical. Existing output files are protected. ``train.predict`` validates
    each checkpoint's encoder hashes before any member probability is used.
    """
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Ensemble output already exists: {output}")
    if split not in {"val", "test"}:
        raise ValueError("Ensemble split must be val or test")
    specs, target_codes = _normalize_members(members)
    patient_ids = _read_ids(specs[0]["prepared_dir"], split)
    for member in specs[1:]:
        if set(_read_ids(member["prepared_dir"], split)) != set(patient_ids):
            raise ValueError("Ensemble prepared datasets have different patient coverage")
    total = np.zeros((len(patient_ids), len(target_codes)), dtype=np.float64)
    with TemporaryDirectory(prefix="ehr_ensemble_predictions_") as temporary:
        for index, member in enumerate(specs):
            member_csv = Path(temporary) / f"member_{index:03d}.csv"
            _predict_member(member["checkpoint"], member["prepared_dir"], member_csv, device, split)
            total += _read_aligned_csv(member_csv, member["target_codes"], patient_ids, target_codes)
    # Recheck after inference to protect an output another caller may have made.
    if output.exists():
        raise FileExistsError(f"Ensemble output already exists: {output}")
    validation = write_predictions(output, patient_ids, total / len(specs), target_codes, patient_ids)
    return {**validation, "n_members": len(specs), "split": split,
            "aggregation": "uniform arithmetic mean of member probabilities",
            "weight_per_member": 1 / len(specs), "target_codes": target_codes,
            "members": [{"checkpoint": str(member["checkpoint"]), "prepared_dir": str(member["prepared_dir"]),
                         "seed": member["config"]["seed"], "model_type": member["config"]["model_type"],
                         "checkpoint_sha256": member["checkpoint_sha256"],
                         "encoder_sha256": member["encoder_sha256"]} for member in specs]}


def _preparation_recipe(member):
    metadata = member["metadata"]
    preprocessing = _json(member["prepared_dir"] / "preprocessing.json")
    kwargs = {"max_events": metadata["max_events"], "min_token_count": metadata.get("min_token_count", 3),
              "n_value_bins": metadata.get("n_value_bins", 8)}
    if preprocessing.get("full_history_summary") or metadata.get("full_history_summary"):
        relative_path = "prepared_summary"
        if metadata.get("summary_standardized_clip") is not None:
            kwargs["standardized_clip"] = metadata["summary_standardized_clip"]
            relative_path += f"_clip{metadata['summary_standardized_clip']:g}"
        return {"mode": "summary", "prepared_relative_path": relative_path, "kwargs": kwargs}
    if preprocessing.get("schema_version", 1) >= 2 and preprocessing.get("vocabulary_scope"):
        return {"mode": "length", "prepared_relative_path": f"prepared_l{metadata['max_events']}", "kwargs": kwargs}
    return {"mode": "original", "prepared_relative_path": "prepared_original", "kwargs": kwargs}


def _saved_validation_metrics(specs, target_codes):
    if not all((member["checkpoint"].parent / "val_predictions.csv").is_file() for member in specs):
        return None
    reference_ids = _read_ids(specs[0]["prepared_dir"], "val")
    with np.load(specs[0]["prepared_dir"] / "val.npz", allow_pickle=False) as arrays:
        own_labels = arrays["labels"]
    labels = own_labels[:, [specs[0]["target_codes"].index(code) for code in target_codes]]
    scores = np.zeros_like(labels, dtype=np.float64)
    for member in specs:
        own_ids = _read_ids(member["prepared_dir"], "val")
        if set(own_ids) != set(reference_ids):
            raise ValueError("Member validation populations differ")
        with np.load(member["prepared_dir"] / "val.npz", allow_pickle=False) as arrays:
            own_y = arrays["labels"]
        order = [own_ids.index(patient) for patient in reference_ids]
        columns = [member["target_codes"].index(code) for code in target_codes]
        if not np.array_equal(own_y[np.ix_(order, columns)], labels):
            raise ValueError("Member validation labels do not describe the same patients/targets")
        scores += _read_aligned_csv(member["checkpoint"].parent / "val_predictions.csv", member["target_codes"],
                                   reference_ids, target_codes)
    return evaluate_predictions(labels, scores / len(specs), target_codes)


def _model_card(entries, validation):
    lines = ["---", "library_name: pytorch", "tags:", "  - pytorch", "  - synthetic-ehr", "  - multi-label-classification", "---", "",
             "# Patient timeline forecasting - equal-weight ensemble", "",
             "This is a local, portable export. No service upload or public submission has been performed.", "",
             f"The prediction is the uniform arithmetic mean of all {len(entries)} declared members. "
             "Weights are fixed, not fitted on test outcomes; no best seed is selected within this manifest. "
             "Each member retains its own fitted encoder, input dimensions and checkpoint.", "", "## Members", ""]
    for index, member in enumerate(entries):
        lines.append(f"- Member {index}: {member['model_type']}, seed {member['seed']}, "
                     f"weight {member['weight']:.8f}, preparation mode {member['preparation_mode']}, "
                     f"max events {member['max_events']}.")
    if validation is not None:
        lines += ["", "## Saved validation prediction metrics", "",
                  f"mAP over all {validation['n_targets']} targets: {validation['mAP']:.6f}. "
                  f"Macro AUROC: {validation['macro_auroc']} over {validation['n_auroc_valid']} eligible targets. "
                  f"Validation patients: {validation['n_patients']}. "
                  "These metrics are recomputed from saved member validation predictions. "
                  "The validation cohort was also used for checkpoint/model selection, so this is not an unbiased test estimate."]
    lines += ["", "## Reproduction", "",
              "Install this directory with `python -m pip install -e .` using a suitable PyTorch environment. "
              "If present, requirements-lock.txt captures the original resolved dependencies.", "",
              "Raw-table reproduction requires the exact official training release as well as the test tables, "
              "because the current preparation functions reconstruct train-fitted encoders. "
              "The frozen vocabulary/preprocessing files and expected SHA256 values are included for every member. "
              "Use `python -m ehr_forecast prepare-ensemble --manifest ensemble.json "
              "--data-dir research_intern_coding_test --output-root artifacts/optimization` to rebuild and verify every recipe. "
              "Alternatively use ensemble.json preparation_mode, preparation_kwargs and prepared_relative_path to rebuild each cache; "
              "the prediction loader rejects encoder mismatches.", "",
              "For common-encoder lengths, run `python -m ehr_forecast prepare-experiment --mode length "
              "--data-dir research_intern_coding_test --output-root artifacts/optimization`. "
              "For full-history summaries use the same command with `--mode summary`. "
              "The original encoder uses `python -m ehr_forecast prepare` and an explicitly chosen output directory. "
              "A manifest with non-default recipe arguments must use those exact arguments; if paths carry a member-specific "
              "subdirectory, prepare into that subdirectory. The manifest records all relevant dimensions and encoder hashes.", "",
              "Inference uses `python -m ehr_forecast predict-ensemble "
              "--manifest ensemble.json --prepared-root artifacts/optimization --output predictions.csv`. "
              "Alternatively call ehr_forecast.ensemble.predict_ensemble with each relative checkpoint resolved beneath "
              "this directory and each prepared_relative_path resolved beneath your cache root.", "",
              "## Representation, task and limitations", "",
              "All weights were trained from scratch on the supplied synthetic training patients. "
              "No external data or pretrained weights were used. Histories are strictly before each anchor; labels "
              "are first-ever diagnoses in the following five-year window. Numeric bins, vocabulary, static scaling "
              "and full-history summary transforms are fitted only on training histories. Known pre-anchor target "
              "conditions receive zero future-incidence probability. Output probabilities are independent across conditions.", "",
              "Different member recipes may retain recent event sequences, full-history count/recency summaries, or both; "
              "consult model_config.json, metadata.json and preprocessing.json for exact member architectures and dimensions. "
              "Rare-condition scores and validation model-selection uncertainty remain limitations. Test outcomes are hidden; "
              "test AUROC/mAP are not reported. No clinical generalization has been established.", "",
              "wandb runs were recorded locally; authenticate and sync the original offline run directories to obtain "
              "reviewer-accessible experiment URLs. Raw/prepared patient data, patient-level predictions, credentials "
              "and wandb payloads are absent from this bundle.", "",
              "## Data attribution and integrity", "",
              "Derived from the Coherent Synthetic Data Set, generated by Synthea, CC BY 4.0 "
              "(https://creativecommons.org/licenses/by/4.0/). Copyright 2022 MITRE, Public Release Case 21-1917. "
              "No source patient records are redistributed.", "",
              "manifest.json hashes every other packaged file and excludes itself to avoid self-reference.", ""]
    return "\n".join(lines)


def export_ensemble(members: list[dict], output_dir: Path, source_root: Path | None = None) -> dict:
    """Package all declared members, frozen encoders and inference sources locally.

    A completed run configuration/summary is required for each checkpoint. No
    patient arrays or prediction CSVs are copied. Existing nonempty destinations
    are protected. Member encoders may differ, and their cache recipes remain
    separate in the portable manifest.
    """
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Export destination must be absent or empty: {output}")
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[1]
    if not (root / "ehr_forecast").is_dir() or not (root / "pyproject.toml").is_file():
        raise FileNotFoundError("Source root must contain ehr_forecast/ and pyproject.toml")
    specs, target_codes = _normalize_members(members)
    required = ("model_config.json", "metadata.json", "vocabulary.json", "preprocessing.json",
                "validation_metrics.json", "summary.json", "environment.json")
    for member in specs:
        run_dir = member["checkpoint"].parent
        for name in required:
            if not (run_dir / name).is_file():
                raise FileNotFoundError(run_dir / name)
        summary = _json(run_dir / "summary.json")
        if summary.get("checkpoint_sha256") != member["checkpoint_sha256"]:
            raise ValueError("Checkpoint does not match the selected completed-run summary SHA256")
        if summary.get("test_labels_available") is True or summary.get("test_metrics") is not None:
            raise ValueError("Hidden test outcomes must not be claimed in the export")
        for name, expected in member["encoder_sha256"].items():
            if _sha256(run_dir / name) != expected:
                raise ValueError(f"Run encoder differs from its explicitly matched prepared dataset: {name}")
        run_metadata = _json(run_dir / "metadata.json")
        if run_metadata["target_codes"] != member["target_codes"]:
            raise ValueError("Run and prepared target ordering differ")
        if any(run_metadata.get(key) != member["metadata"].get(key) for key in ("max_events", "n_static", "vocab_size")):
            raise ValueError("Run and prepared input dimensions differ")
    validation = _saved_validation_metrics(specs, target_codes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ehr_ensemble_export_", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        entries, assigned_paths = [], {}
        for index, member in enumerate(specs):
            relative = Path("members") / f"member_{index:03d}"
            destination = stage / relative
            destination.mkdir(parents=True)
            shutil.copy2(member["checkpoint"], destination / "model.pt")
            for name in required:
                shutil.copy2(member["checkpoint"].parent / name, destination / name)
            shutil.copy2(member["config_path"], destination / "training_config.json")
            recipe = _preparation_recipe(member)
            signature = json.dumps({"encoder": member["encoder_sha256"], "kwargs": recipe["kwargs"]}, sort_keys=True)
            prepared_relative = recipe["prepared_relative_path"]
            if prepared_relative in assigned_paths and assigned_paths[prepared_relative] != signature:
                prepared_relative = f"member_{index:03d}/{prepared_relative}"
            assigned_paths[prepared_relative] = signature
            entries.append({"checkpoint": (relative / "model.pt").as_posix(),
                            "training_config": (relative / "training_config.json").as_posix(),
                            "seed": member["config"]["seed"], "model_type": member["config"]["model_type"],
                            "weight": 1 / len(specs), "max_events": member["metadata"]["max_events"],
                            "n_static": member["metadata"].get("n_static"),
                            "preparation_mode": recipe["mode"], "preparation_kwargs": recipe["kwargs"],
                            "prepared_relative_path": prepared_relative,
                            "checkpoint_sha256": member["checkpoint_sha256"], "encoder_sha256": member["encoder_sha256"]})
        shutil.copy2(root / "pyproject.toml", stage / "pyproject.toml")
        for source in sorted((root / "ehr_forecast").rglob("*.py")):
            if "__pycache__" not in source.parts:
                target = stage / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        lock = root / "artifacts" / "environment-lock.txt"
        if lock.is_file():
            shutil.copy2(lock, stage / "requirements-lock.txt")
        payload = {"schema_version": 1, "aggregation": "equal_weight_probability_mean", "target_codes": target_codes,
                   "n_members": len(entries), "members": entries, "test_metrics": None, "uploaded": False,
                   "validation_metrics_source": "saved member validation predictions" if validation else None,
                   "prepared_root_convention": "Resolve each prepared_relative_path beneath a user-supplied prepared root"}
        _write_json(stage / "ensemble.json", payload)
        if validation:
            _write_json(stage / "validation_metrics.json", validation)
        (stage / "README.md").write_text(_model_card(entries, validation), encoding="utf-8")
        files = {path.relative_to(stage).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(stage.rglob("*")) if path.is_file()}
        _write_json(stage / "manifest.json", {"schema_version": 1, "files": files, "uploaded": False,
                                               "manifest_policy": "Hashes cover every other packaged file; this manifest excludes itself."})
        if output.exists():
            output.rmdir()  # Only an empty destination is permitted; a concurrent file prevents removal.
        stage.rename(output)
    return {"output_dir": str(output), "n_members": len(entries), "files": len(files) + 1,
            "aggregation": "equal_weight_probability_mean", "validation_mAP": validation["mAP"] if validation else None,
            "test_metrics": None, "uploaded": False}
