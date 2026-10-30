"""Frozen, strict-deterministic A/A replication before the declared 2x2 audit.

The original warn-only runs and their failed gate are immutable background.
Only repeat-a enters the factorial; repeat-b is never selected by its score.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_experimental_audit as original
from ehr_forecast.experiment_analysis import load_aligned_predictions
from ehr_forecast.train import sha256_file, state_fingerprint, train, write_json

AUDIT = ROOT / "artifacts/experimental_audit_strict"
PREPARED = original.PREPARED
SEEDS = original.SEEDS
CELLS = original.CELLS
ORIGINAL_MANIFEST = original.AUDIT / "frozen_manifest.json"
ORIGINAL_PROTOCOL = ROOT / "docs/experimental_protocol.md"
STRICT_PROTOCOL = ROOT / "docs/experimental_protocol_strict.md"
UPSTREAM_GATES = ("gate_a_reload.json", "gate_b_equivalence.json")
TIME_FIELDS = frozenset({"elapsed_seconds"})
ENVIRONMENT_FIELDS = (
    "python", "platform", "torch", "numpy", "cuda_runtime", "device", "gpu",
    "deterministic_algorithms", "deterministic_enabled", "deterministic_warn_only",
    "cublas_workspace_config", "cudnn_benchmark", "cudnn_deterministic",
    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision",
    "torch_num_threads", "amp_enabled", "amp_dtype", "sdp_backends_enabled",
)
REQUIRED_RUN_FILES = (
    "config.json", "model_config.json", "environment.json", "prepared_sha256.json",
    "initial_state.json", "initial.pt", "training_begin_rng.json", "history.jsonl",
    "best.pt", "summary.json", "val_predictions.csv", "test_predictions.csv",
    "validation_metrics.json", "supervised_parameter_audit.json", "strict_provenance.json",
)
REQUIRED_HISTORY_FIELDS = frozenset({
    "stage", "epoch", "train/loss", "val/mAP", "val/macro_auroc", "val/loss",
    "learning_rate", "train_eval/BCE", "train_eval/mAP", "train_eval/macro_auroc",
    "train_eval/Brier", "gradient/norm_mean", "gradient/norm_max",
    "gradient/clip_fraction", "amp/skipped_steps", "train_patient_order_sha256",
    "val_long/mAP", "val_long/macro_auroc", "val_short/mAP", "val_short/macro_auroc",
})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def source_paths():
    """Freeze all runtime modules and the runners, rather than mutable reports."""
    paths = list((ROOT / "ehr_forecast").glob("*.py"))
    paths += [ROOT / name for name in (
        "scripts/run_experimental_audit.py", "scripts/run_strict_audit.py",
        "scripts/check_audit_equivalence.py", "pyproject.toml",
    ) if (ROOT / name).is_file()]
    return sorted(paths)


def upstream():
    manifest = original.verify_frozen()
    if sha256_file(ORIGINAL_PROTOCOL) != manifest["protocol_sha256"]:
        raise ValueError("Original protocol changed after its freeze")
    for name in UPSTREAM_GATES:
        gate = read(original.AUDIT / name)
        if gate.get("passed") is not True:
            raise RuntimeError(f"Original prerequisite {name} must pass")
        rows = gate.get("runs") if name == "gate_a_reload.json" else gate.get("seeds")
        if not rows or sorted(row["seed"] for row in rows) != SEEDS:
            raise ValueError(f"Original prerequisite {name} does not cover all declared seeds")
    return manifest


def frozen_candidate():
    previous = upstream()
    if {path.name: sha256_file(path) for path in PREPARED.iterdir() if path.is_file()} != previous["prepared_sha256"]:
        raise ValueError("Prepared input file set or fingerprints changed from the original freeze")
    return {
        "schema_version": 1, "deterministic_mode": "strict", "seeds": SEEDS,
        "prepared_dir": str(PREPARED), "prepared_sha256": previous["prepared_sha256"],
        "original_frozen_manifest_path": relative(ORIGINAL_MANIFEST),
        "original_frozen_manifest_sha256": sha256_file(ORIGINAL_MANIFEST),
        "original_protocol_path": relative(ORIGINAL_PROTOCOL),
        "original_protocol_sha256": sha256_file(ORIGINAL_PROTOCOL),
        "strict_protocol_path": relative(STRICT_PROTOCOL),
        "strict_protocol_sha256": sha256_file(STRICT_PROTOCOL),
        "source_sha256": {relative(path): sha256_file(path) for path in source_paths()},
        "auxiliary_inputs_sha256": {"artifacts/optimization/history_patient_groups.csv":
                                     sha256_file(ROOT / "artifacts/optimization/history_patient_groups.csv")},
        "upstream_gates_sha256": {name: sha256_file(original.AUDIT / name) for name in UPSTREAM_GATES},
        "baseline_rule": "C0R0 repeat-a, regardless of the repeat-b score",
        "comparison_scope": "Same software, hardware and configuration; exact non-time A/A equality",
    }


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Refusing to overwrite different existing evidence: {path}")
    else:
        write_json(path, value)


def freeze():
    manifest = frozen_candidate()
    immutable_json(AUDIT / "frozen_manifest.json", manifest)
    for name, fingerprint in manifest["source_sha256"].items():
        destination = AUDIT / "source_snapshot" / name
        if destination.exists():
            if sha256_file(destination) != fingerprint:
                raise ValueError(f"Source snapshot changed: {name}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
    return verify_frozen()


def verify_frozen():
    manifest = read(AUDIT / "frozen_manifest.json")
    if manifest != frozen_candidate():
        raise ValueError("Strict frozen sources, protocols, inputs or upstream gates changed; stop")
    for name, fingerprint in manifest["source_sha256"].items():
        if sha256_file(AUDIT / "source_snapshot" / name) != fingerprint:
            raise ValueError(f"Strict source snapshot fingerprint changed: {name}")
    return manifest


def config_for(cell, seed, repeat=None):
    if cell not in CELLS or seed not in SEEDS:
        raise ValueError("Only declared factorial cells and seeds are allowed")
    if (cell == "C0R0" and repeat not in ("a", "b")) or (cell != "C0R0" and repeat is not None):
        raise ValueError("C0R0 requires repeat-a or repeat-b; other cells have no repeat")
    config = original.config_for(cell, seed)
    suffix = f"{cell}-seed{seed}" + (f"-repeat-{repeat}" if repeat else "")
    config.update(name=f"strict-audit-{suffix}", deterministic_mode="strict",
                  output_dir=relative(AUDIT / "runs" / suffix))
    return config


def provenance(manifest):
    return {
        "schema_version": 1, "deterministic_mode": "strict",
        "frozen_manifest_path": relative(AUDIT / "frozen_manifest.json"),
        "frozen_manifest_sha256": sha256_file(AUDIT / "frozen_manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "strict_protocol_sha256": manifest["strict_protocol_sha256"],
        "original_frozen_manifest_sha256": manifest["original_frozen_manifest_sha256"],
        "upstream_gates_sha256": manifest["upstream_gates_sha256"],
    }


def validate_run(directory, config, manifest):
    missing = [name for name in REQUIRED_RUN_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete strict run {directory}: {missing}")
    if read(directory / "config.json") != config:
        raise ValueError(f"Strict run configuration differs: {directory}")
    if read(directory / "strict_provenance.json") != provenance(manifest):
        raise ValueError(f"Strict run provenance differs: {directory}")
    if read(directory / "prepared_sha256.json") != manifest["prepared_sha256"]:
        raise ValueError(f"Strict run prepared inputs differ: {directory}")
    environment = read(directory / "environment.json")
    if (not set(ENVIRONMENT_FIELDS).issubset(environment)
            or environment["deterministic_algorithms"] != "enabled_strict"
            or environment["deterministic_enabled"] is not True
            or environment["deterministic_warn_only"] is not False
            or environment["cudnn_benchmark"] is not False
            or environment["cudnn_deterministic"] is not True):
        raise ValueError(f"Actual strict determinism was not recorded: {directory}")
    summary = read(directory / "summary.json")
    if summary["checkpoint_sha256"] != sha256_file(directory / "best.pt"):
        raise ValueError(f"Best checkpoint differs from saved summary: {directory}")
    return environment


def run_one(cell, seed, repeat=None):
    manifest = verify_frozen()
    config = config_for(cell, seed, repeat)
    directory = ROOT / config["output_dir"]
    immutable_json(AUDIT / "configs" / f"{directory.name}.json", config)
    if (directory / "summary.json").exists():
        validate_run(directory, config, manifest)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Partial run exists; refusing to overwrite: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "strict_provenance.json", provenance(manifest))
        print(f"BEGIN strict {directory.name}", flush=True)
        train(config)
        verify_frozen()
        validate_run(directory, config, manifest)
    return directory


def history(directory):
    rows = [json.loads(line) for line in (directory / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not REQUIRED_HISTORY_FIELDS.issubset(row) or row["stage"] != "finetune" for row in rows):
        raise ValueError(f"Complete finetune diagnostics missing: {directory}")
    if [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Nonconsecutive epoch history: {directory}")
    return [{key: value for key, value in row.items() if key not in TIME_FIELDS} for row in rows]


def checkpoint_state(directory, filename):
    # CPU-only fingerprinting; archive byte hashes may differ due to run names/configs.
    checkpoint = torch.load(directory / filename, map_location="cpu", weights_only=True)
    return state_fingerprint(checkpoint["model_state_dict"]), checkpoint


def prediction_comparison(first, second, split):
    with np.load(PREPARED / f"{split}.npz", allow_pickle=False) as arrays:
        ids = arrays["patient_ids"].astype(str).tolist()
    codes = read(PREPARED / "metadata.json")["target_codes"]
    a = load_aligned_predictions(first / f"{split}_predictions.csv", ids, codes)
    b = load_aligned_predictions(second / f"{split}_predictions.csv", ids, codes)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return {"identical": finite and bool(np.array_equal(a, b)), "finite": finite,
            "max_absolute_difference": float(np.max(np.abs(a - b))), "shape": list(a.shape)}


def evidence_hashes(directories):
    return {relative(directory / name): sha256_file(directory / name)
            for directory in directories for name in REQUIRED_RUN_FILES}


def compare_repeats(seed):
    manifest = verify_frozen()
    first = ROOT / config_for("C0R0", seed, "a")["output_dir"]
    second = ROOT / config_for("C0R0", seed, "b")["output_dir"]
    environments = [validate_run(path, config_for("C0R0", seed, repeat), manifest)
                    for path, repeat in ((first, "a"), (second, "b"))]
    configs = [{key: value for key, value in read(path / "config.json").items()
                if key not in ("name", "output_dir")} for path in (first, second)]
    initials = [read(path / "initial_state.json") for path in (first, second)]
    initial_states = [checkpoint_state(path, "initial.pt")[0] for path in (first, second)]
    starts = [read(path / "training_begin_rng.json") for path in (first, second)]
    rows = [history(path) for path in (first, second)]
    summaries = [read(path / "summary.json") for path in (first, second)]
    best = [checkpoint_state(path, "best.pt") for path in (first, second)]
    predictions = {split: prediction_comparison(first, second, split) for split in ("val", "test")}
    checks = {
        "configuration_equal_except_name_output": configs[0] == configs[1],
        "initial_state_equal": (initial_states[0] == initial_states[1]
                                == initials[0]["state_sha256"] == initials[1]["state_sha256"]),
        "initial_rng_equal": initials[0]["rng"] == initials[1]["rng"],
        "initial_metadata_equal": initials[0] == initials[1],
        "training_begin_rng_equal": starts[0] == starts[1],
        "patient_order_equal": ([row["train_patient_order_sha256"] for row in rows[0]]
                                == [row["train_patient_order_sha256"] for row in rows[1]]),
        "epoch_metrics_equal": rows[0] == rows[1],
        "best_epoch_equal": summaries[0]["best_epoch"] == summaries[1]["best_epoch"]
                            == best[0][1]["epoch"] == best[1][1]["epoch"],
        "best_validation_mAP_equal": (summaries[0]["best_validation_mAP"] == summaries[1]["best_validation_mAP"]
                                      == best[0][1]["validation_mAP"] == best[1][1]["validation_mAP"]),
        "best_model_state_equal": best[0][0] == best[1][0],
        "val_predictions_equal": predictions["val"]["identical"],
        "test_predictions_equal": predictions["test"]["identical"],
        "strict_environment_recorded": all(env["deterministic_algorithms"] == "enabled_strict"
                                           and env["deterministic_enabled"] is True
                                           and env["deterministic_warn_only"] is False for env in environments),
        "environment_equal": environments[0] == environments[1],
        "model_configuration_equal": read(first / "model_config.json") == read(second / "model_config.json"),
    }
    mismatches = [{"epoch_a": a.get("epoch"), "epoch_b": b.get("epoch"),
                   "fields": sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))}
                  for a, b in zip(rows[0], rows[1]) if a != b]
    report = {
        "schema_version": 1, "seed": seed, "passed": all(checks.values()),
        "deterministic_mode": "strict", "baseline_run_dir": relative(first),
        "repeat_run_dir": relative(second), "checks": checks,
        "details": {"epoch_counts": [len(value) for value in rows], "epoch_mismatches": mismatches,
                    "best_epochs": [value["best_epoch"] for value in summaries],
                    "best_validation_mAP": [value["best_validation_mAP"] for value in summaries],
                    "initial_state_sha256": initial_states, "best_model_state_sha256": [value[0] for value in best],
                    "predictions": predictions, "excluded_history_fields": sorted(TIME_FIELDS)},
        "frozen_manifest_sha256": sha256_file(AUDIT / "frozen_manifest.json"),
        "artifact_sha256": evidence_hashes((first, second)),
        "scope": "Exact same-environment A/A; repeat-a is the fixed factorial baseline, not the better score",
    }
    immutable_json(AUDIT / f"gate_a_strict_seed{seed}.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Strict A/A seed{seed} failed; stop before factorial attribution")
    return report


def verify_strict_gate(seed):
    report = read(AUDIT / f"gate_a_strict_seed{seed}.json")
    if report.get("passed") is not True or not report.get("checks") or not all(value is True for value in report["checks"].values()):
        raise RuntimeError(f"Strict A/A seed{seed} must pass before factorial attribution")
    if report["frozen_manifest_sha256"] != sha256_file(AUDIT / "frozen_manifest.json"):
        raise ValueError("Strict A/A gate belongs to a different freeze")
    for name, fingerprint in report["artifact_sha256"].items():
        if sha256_file(ROOT / name) != fingerprint:
            raise ValueError(f"Strict A/A evidence changed: {name}")
    # Recompute the checks, rather than trusting only a saved passed flag.
    return compare_repeats(seed)


def release_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "replay", "factorial"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in SEEDS for seed in args.seeds):
        parser.error("Use distinct seeds from the declared 17, 42, 73")
    if args.stage == "freeze":
        result = freeze()
    elif args.stage == "replay":
        verify_frozen()
        result = []
        for seed in args.seeds:
            for repeat in ("a", "b"):
                run_one("C0R0", seed, repeat)
                release_cache()
            result.append(compare_repeats(seed))
    else:
        verify_frozen()
        if args.seeds != SEEDS:
            parser.error("Factorial uses all three declared seeds in the fixed order")
        for seed in SEEDS:
            verify_strict_gate(seed)
        result = []
        for cell in ("C1R0", "C0R1", "C1R1"):
            for seed in SEEDS:
                directory = run_one(cell, seed)
                # Same runtime/precision across cells, apart from their parameter counts.
                baseline = ROOT / config_for("C0R0", seed, "a")["output_dir"]
                current_env, baseline_env = read(directory / "environment.json"), read(baseline / "environment.json")
                if any(current_env[key] != baseline_env[key] for key in ENVIRONMENT_FIELDS):
                    raise ValueError("Factorial runtime or precision changed from the strict baseline")
                result.append({"cell": cell, "seed": seed, "run_dir": relative(directory), **read(directory / "summary.json")})
                release_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
