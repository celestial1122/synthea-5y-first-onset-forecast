"""Freeze and run matched frozen-feature ranking-objective experiments."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranking_study.trainer import METHODS, file_sha, load_feature_cache, read_json, require, tensor_sha, train_head, write_json

STUDY = ROOT / "artifacts/group_ranking"
OLD_MANIFEST = ROOT / "artifacts/experimental_audit_strict/frozen_manifest.json"
SEEDS = (17, 42, 73)
SPEC = {"methods": list(METHODS), "seeds": list(SEEDS), "epochs": 100, "patience": 15,
        "learning_rate": .003, "optimizer": "Adam", "weight_decay": 0., "ridge_mu": .1,
        "kappa": .25, "group_size": 8, "rl_calibration_draws": 4, "chunk_size": 128,
        "training_rng_seed_offset": 17041, "calibration_rng_seed_offset": 27041,
        "temperature_recipe": ".5 * population std(initial train eligible logits), clamp [.25,2]",
        "training_pool": "full training patients, per-disease eligible masks, fixed K macro means",
        "selection": "val mAP, includes epoch0, min improvement1e-6, patience15; fixed epoch10 diagnostic",
        "amp": False, "dropout": False, "dtype": "float32", "deterministic_mode": "strict"}
SOURCES = ("ranking_study/__init__.py", "ranking_study/objectives.py", "ranking_study/trainer.py",
           "scripts/extract_ranking_features.py", "scripts/run_group_ranking.py", "docs/group_ranking_protocol.md")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def frozen_candidate():
    old = read_json(OLD_MANIFEST)
    # Verify the old manifest's explicit file list. New package files cannot
    # retroactively change what the previous study froze.
    for name, expected in old["source_sha256"].items():
        require(file_sha(ROOT / name) == expected, f"Previous frozen source changed: {name}")
    for name, expected in old["prepared_sha256"].items():
        require(file_sha(ROOT / "artifacts/optimization/prepared_l512" / name) == expected,
                f"Previous frozen data changed: {name}")
    cache_hashes = {}
    for seed in SEEDS:
        directory = STUDY / "features" / f"seed{seed}"
        manifest, reference, _, head, arrays = load_feature_cache(directory, ROOT)
        require(manifest["seed"] == seed and tuple(head["weight"].shape) == (40, 133), "Unexpected feature dimensions/seed")
        require([len(arrays[s]["patient_ids"]) for s in ("train", "val", "test")] == [2791, 365, 358],
                "Official split sizes changed")
        require(int(arrays["val"]["history_group_long"].sum()) == 133, "Long-history cohort changed")
        cache_hashes[relative(directory / "manifest.json")] = file_sha(directory / "manifest.json")
        cache_hashes[reference["path"]] = reference["sha256"]
        cache_hashes.update({relative(directory / name): value for name, value in manifest["files_sha256"].items()})
    return {"schema_version": 1, "study": "frozen causal encoder, matched head objectives",
            "specification": SPEC, "previous_manifest_path": relative(OLD_MANIFEST),
            "previous_manifest_sha256": file_sha(OLD_MANIFEST),
            "source_sha256": {name: file_sha(ROOT / name) for name in SOURCES},
            "feature_and_reference_sha256": cache_hashes,
            "inference_scope": "Fixed fitted encoders and repeated validation selection; no end-to-end-model or unseen-test guarantee."}


def freeze():
    candidate = frozen_candidate()
    path = STUDY / "frozen_manifest.json"
    if path.exists():
        require(read_json(path) == candidate, "Refusing to replace a different study freeze")
    else:
        write_json(path, candidate)
    for name, fingerprint in candidate["source_sha256"].items():
        snapshot = STUDY / "source_snapshot" / name
        if snapshot.exists():
            require(file_sha(snapshot) == fingerprint, f"Study source snapshot changed: {name}")
        else:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, snapshot)
    return candidate


def verify():
    candidate = frozen_candidate()
    require(read_json(STUDY / "frozen_manifest.json") == candidate, "Study source/input/protocol drifted after freeze")
    for name, fingerprint in candidate["source_sha256"].items():
        require(file_sha(STUDY / "source_snapshot" / name) == fingerprint, f"Study source snapshot changed: {name}")
    return candidate


def config_for(method, seed, device, repeat=False):
    require(method in METHODS and seed in SEEDS, "Undeclared method/seed")
    return {**SPEC, "method": method, "seed": seed, "device": device,
            "feature_dir": relative(STUDY / "features" / f"seed{seed}"),
            "output_dir": relative(STUDY / "runs" / (f"{method}-seed{seed}" + ("-repeat" if repeat else ""))),
            "wandb_mode": "offline", "wandb_project": "patient-timeline-forecast"}


def run_one(method, seed, device, repeat=False):
    manifest = verify()
    config = config_for(method, seed, device, repeat)
    output = ROOT / config["output_dir"]
    provenance = {"frozen_manifest_path": relative(STUDY / "frozen_manifest.json"),
                  "frozen_manifest_sha256": file_sha(STUDY / "frozen_manifest.json"),
                  "source_sha256": manifest["source_sha256"],
                  "feature_manifest_sha256": file_sha(ROOT / config["feature_dir"] / "manifest.json")}
    if (output / "summary.json").exists():
        require(read_json(output / "config.json") == config and read_json(output / "provenance.json") == provenance,
                "Existing run has different configuration/provenance")
        summary = read_json(output / "summary.json")
        require(summary["checkpoint_sha256"] == file_sha(output / "best.pt"), "Saved checkpoint changed")
        require((output / "artifact_sha256.json").is_file(), "Completed run lacks its artifact integrity record")
        for name, fingerprint in read_json(output / "artifact_sha256.json").items():
            require(file_sha(output / name) == fingerprint, f"Saved run evidence changed: {name}")
        return summary
    require(not output.exists() or not any(output.iterdir()), "Partial run exists; refusing to overwrite evidence")
    import wandb
    output.mkdir(parents=True, exist_ok=True)
    # Wandb writes into its own sibling directory; train_head's output remains
    # empty until it has validated and recorded the actual run.
    tracking = STUDY / "tracking" / (f"{method}-seed{seed}" + ("-repeat" if repeat else ""))
    tracking.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=config["wandb_project"], name=f"group-ranking-{tracking.name}",
                     mode="offline", dir=str(tracking.resolve()), config=config,
                     settings=wandb.Settings(init_timeout=90))
    try:
        result = train_head(config, root=ROOT, provenance=provenance, tracker=run)
        write_json(output / "tracking.json", {"id": run.id, "mode": "offline", "dir": run.dir})
        verify()
        write_json(output / "artifact_sha256.json", {path.name: file_sha(path) for path in sorted(output.iterdir())
                                                     if path.is_file() and path.name != "artifact_sha256.json"})
        return result
    finally:
        run.finish()


def compare_repeat(method, seed=17):
    """Exact same-environment A/A, with no score-based choice of a repeat."""
    import json
    import torch
    first, second = [STUDY / "runs" / (f"{method}-seed{seed}" + suffix) for suffix in ("", "-repeat")]
    def rows(path):
        return [{key: value for key, value in json.loads(line).items() if key != "elapsed_seconds"}
                for line in (path / "history.jsonl").read_text().splitlines()]
    def state(path, name):
        return tensor_sha(torch.load(path / name, map_location="cpu", weights_only=True))
    configs = [{k: v for k, v in read_json(path / "config.json").items() if k != "output_dir"} for path in (first, second)]
    summaries = [{k: v for k, v in read_json(path / "summary.json").items()
                  if k not in {"elapsed_seconds", "checkpoint_sha256"}} for path in (first, second)]
    evidence = {}
    for path in (first, second):
        for name, fingerprint in read_json(path / "artifact_sha256.json").items():
            require(file_sha(path / name) == fingerprint, f"A/A evidence changed: {path / name}")
            evidence[relative(path / name)] = fingerprint
    checks = {"configuration_equal_except_output": configs[0] == configs[1],
              "initialization_equal": read_json(first / "initialization.json") == read_json(second / "initialization.json"),
              "provenance_equal": read_json(first / "provenance.json") == read_json(second / "provenance.json"),
              "environment_equal": read_json(first / "environment.json") == read_json(second / "environment.json"),
              "initial_head_equal": state(first, "initial_head.pt") == state(second, "initial_head.pt"),
              "history_non_time_equal": rows(first) == rows(second),
              "best_head_equal": state(first, "best_head.pt") == state(second, "best_head.pt"),
              "epoch10_head_equal": state(first, "epoch10_head.pt") == state(second, "epoch10_head.pt"),
              "summary_non_time_equal": summaries[0] == summaries[1]}
    for name in ("val_predictions.csv", "test_predictions.csv", "train_predictions.csv", "epoch10_val_predictions.csv"):
        checks[f"{name}_equal"] = (first / name).read_bytes() == (second / name).read_bytes()
    result = {"passed": all(checks.values()), "method": method, "seed": seed, "checks": checks,
              "baseline_run_dir": relative(first), "repeat_run_dir": relative(second),
              "artifact_sha256": evidence,
              "selection_note": "The repeat is only a reproducibility check and never replaces the primary run."}
    path = STUDY / f"aa_{method}_seed{seed}.json"
    if path.exists():
        require(read_json(path) == result, "Refusing to replace different A/A evidence")
    else:
        write_json(path, result)
    require(result["passed"], "Exact A/A failed; stop attribution, do not relax the tolerance")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "verify", "train", "repeat"), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    require(len(set(args.methods)) == len(args.methods) and len(set(args.seeds)) == len(args.seeds), "Duplicate method/seed")
    if args.stage == "freeze":
        freeze()
        print("Ranking study inputs/protocol/source frozen")
    elif args.stage == "verify":
        verify()
        print("Ranking study freeze intact")
    elif args.stage == "train":
        require(args.device == "cuda", "Declared full study requires CUDA; use train_head directly for synthetic CPU smoke")
        import torch
        for method in args.methods:
            for seed in args.seeds:
                result = run_one(method, seed, args.device)
                print({key: result[key] for key in ("method", "seed", "best_epoch", "best_validation_mAP", "no_update_selected")}, flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    else:
        require(args.device == "cuda", "Declared A/A requires the same CUDA condition")
        verify()
        import torch
        for method in ("bce", "grpo_ap"):
            # Check the primary run/provenance before launching its repeat.
            require((STUDY / "runs" / f"{method}-seed17" / "summary.json").exists(), "Primary seed17 fit must complete first")
            run_one(method, 17, args.device)
            run_one(method, 17, args.device, repeat=True)
            print(compare_repeat(method), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
