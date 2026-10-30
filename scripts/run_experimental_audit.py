"""Gated reproduction and small factorial attribution; no architecture search."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ehr_forecast.experiment_analysis import load_aligned_predictions
from ehr_forecast.metrics import evaluate_predictions
from ehr_forecast.train import predict, sha256_file, train, write_json

AUDIT = ROOT / "artifacts/experimental_audit"
PREPARED = ROOT / "artifacts/optimization/prepared_l512"
SEEDS = [17, 42, 73]
CELLS = {"C0R0": ("causal", "legacy"), "C1R0": ("bidirectional", "legacy"),
         "C0R1": ("causal", "pooled_mlp"), "C1R1": ("bidirectional", "pooled_mlp")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference(seed):
    return ROOT / f"artifacts/optimization/runs/no-pretrain-l512-seed{seed}"


def legacy_ast(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pieces = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
              and node.name in {"TimelineTransformer", "_validate_shapes"}]
    return ast.dump(ast.Module(body=pieces, type_ignores=[]), include_attributes=False)


def freeze():
    AUDIT.mkdir(parents=True, exist_ok=True)
    files = {path.name: sha256_file(path) for path in PREPARED.iterdir() if path.is_file()}
    legacy_snapshot = ROOT / "artifacts/optimized_model_bundle/ehr_forecast/models.py"
    same_legacy = legacy_ast(legacy_snapshot) == legacy_ast(ROOT / "ehr_forecast/models.py")
    if not same_legacy:
        raise ValueError("Legacy model definition changed from the preserved previous export")
    manifest = {"seeds": SEEDS, "prepared_dir": str(PREPARED), "prepared_sha256": files,
                "protocol_sha256": sha256_file(ROOT / "docs/experimental_protocol.md"),
                "legacy_model_ast_matches_previous_export": same_legacy,
                "references": {str(seed): {name: sha256_file(reference(seed) / name)
                                           for name in ("best.pt", "config.json", "metadata.json", "val_predictions.csv")}
                               for seed in SEEDS}}
    path = AUDIT / "frozen_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Frozen audit inputs changed; do not silently replace the protocol or inputs")
    write_json(path, manifest)
    source = AUDIT / "source_snapshot"
    source.mkdir(exist_ok=True)
    for name in ("models.py", "train.py", "metrics.py", "data.py"):
        destination = source / name
        if not destination.exists():
            shutil.copy2(ROOT / "ehr_forecast" / name, destination)
    return manifest


def verify_frozen():
    manifest = read(AUDIT / "frozen_manifest.json")
    if any(sha256_file(PREPARED / name) != value for name, value in manifest["prepared_sha256"].items()):
        raise ValueError("Prepared input fingerprint changed")
    for seed in SEEDS:
        for name, expected in manifest["references"][str(seed)].items():
            if sha256_file(reference(seed) / name) != expected:
                raise ValueError("Reference artifact fingerprint changed")
    return manifest


def reload_gate():
    verify_frozen()
    metadata = read(PREPARED / "metadata.json")
    with np.load(PREPARED / "val.npz", allow_pickle=False) as arrays:
        ids, labels = arrays["patient_ids"].astype(str).tolist(), arrays["labels"].copy()
    results = []
    for seed in SEEDS:
        destination = AUDIT / f"reloaded-seed{seed}.csv"
        predict(reference(seed) / "best.pt", PREPARED, destination, split="val")
        old = load_aligned_predictions(reference(seed) / "val_predictions.csv", ids, metadata["target_codes"])
        new = load_aligned_predictions(destination, ids, metadata["target_codes"])
        maximum = float(np.max(np.abs(new - old)))
        metrics = evaluate_predictions(labels, new, metadata["target_codes"])
        saved = read(reference(seed) / "validation_metrics.json")
        delta = metrics["mAP"] - saved["mAP"]
        results.append({"seed": seed, "max_probability_difference": maximum,
                        "mAP_difference": delta, "passed": maximum <= 1e-6 and abs(delta) <= 1e-10})
    report = {"passed": all(row["passed"] for row in results), "runs": results,
              "scope": "Reloaded saved checkpoints; this is not a fresh-training reproducibility claim"}
    write_json(AUDIT / "gate_a_reload.json", report)
    if not report["passed"]:
        raise RuntimeError("Checkpoint reload gate failed")
    return report


def config_for(cell, seed):
    config = dict(read(reference(seed) / "config.json"))
    config.update(name=f"audit-{cell}-seed{seed}", output_dir=f"artifacts/experimental_audit/runs/{cell}-seed{seed}",
                  audit_diagnostics=True, validation_groups_csv="artifacts/optimization/history_patient_groups.csv")
    if cell != "C0R0":
        config.update(model_type="experimental_bridge", attention_mode=CELLS[cell][0], readout_mode=CELLS[cell][1])
    return config


def replay_comparison(seed):
    directory = AUDIT / f"runs/C0R0-seed{seed}"
    current, previous = read(directory / "summary.json"), read(reference(seed) / "summary.json")
    rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    old = [json.loads(line) for line in (reference(seed) / "history.jsonl").read_text().splitlines()]
    new_by_epoch = {row["epoch"]: row for row in rows if row["stage"] == "finetune"}
    differences = [{"epoch": row["epoch"], "mAP_difference": new_by_epoch[row["epoch"]]["val/mAP"] - row["val/mAP"],
                    "training_online_BCE_difference": new_by_epoch[row["epoch"]]["train/loss"] - row["train/loss"]}
                   for row in old if row["stage"] == "finetune" and row["epoch"] in new_by_epoch]
    best_delta = current["best_validation_mAP"] - previous["best_validation_mAP"]
    maximum = max(abs(row["mAP_difference"]) for row in differences)
    result = {"seed": seed, "passed": abs(best_delta) <= .002 and maximum <= .005,
              "best_mAP_difference": best_delta, "max_fixed_epoch_mAP_difference": maximum,
              "fixed_epoch_differences": differences, "legacy_best_epoch": previous["best_epoch"],
              "replayed_best_epoch": current["best_epoch"],
              "caveat": "Warn-only CUDA training may be nondeterministic; thresholds are audit alarms, not statistical significance"}
    write_json(AUDIT / f"gate_a_replay_seed{seed}.json", result)
    if not result["passed"]:
        raise RuntimeError("A/A fresh-training gate failed; stop attribution and diagnose reproducibility")
    return result


def run_one(cell, seed):
    config = config_for(cell, seed)
    write_json(ROOT / "configs/experimental_audit" / f"{cell}-seed{seed}.json", config)
    directory = ROOT / config["output_dir"]
    if (directory / "summary.json").exists():
        if read(directory / "config.json") != config:
            raise ValueError("Existing audit run configuration differs")
    else:
        print(f"BEGIN {cell} seed{seed}", flush=True)
        train(config)
    if cell == "C0R0":
        return replay_comparison(seed)
    return read(directory / "summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "reload", "replay", "factorial"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze()
    elif args.stage == "reload":
        result = reload_gate()
    elif args.stage == "replay":
        verify_frozen()
        if not read(AUDIT / "gate_a_reload.json")["passed"]:
            raise RuntimeError("Reload gate must pass before a fresh fit")
        result = []
        for seed in args.seeds:
            if seed not in SEEDS:
                raise ValueError("Only declared seeds are allowed")
            if seed != 17 and not read(AUDIT / "gate_a_replay_seed17.json")["passed"]:
                raise RuntimeError("Seed17 reproduction must pass first")
            result.append(run_one("C0R0", seed))
    else:
        verify_frozen()
        if not read(AUDIT / "gate_b_equivalence.json")["passed"]:
            raise RuntimeError("Pipeline equivalence gate must pass before attribution")
        for seed in SEEDS:
            if not read(AUDIT / f"gate_a_replay_seed{seed}.json")["passed"]:
                raise RuntimeError("All three baseline reproductions must pass")
        result = []
        import gc
        import torch
        for cell in ("C1R0", "C0R1", "C1R1"):
            for seed in SEEDS:
                result.append({"cell": cell, "seed": seed, **run_one(cell, seed)})
                gc.collect()
                torch.cuda.empty_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
