"""Freeze and run the single scale-invariant ranking refinement."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranking_study.scale_invariant_trainer import METHODS, file_sha, read_json, require, tensor_sha, train_head, write_json
from scripts.run_group_ranking import verify as verify_v1

STUDY = ROOT / "artifacts/group_ranking_scale_invariant"
V1 = ROOT / "artifacts/group_ranking"
FEATURES = V1 / "features"
OLD_MANIFEST = V1 / "frozen_manifest.json"
DIAGNOSTICS = V1 / "gradient_diagnostics.json"
SEEDS = (17, 42, 73)
SPEC = {"methods": list(METHODS), "seeds": list(SEEDS), "epochs": 100, "patience": 15,
        "learning_rate": .003, "optimizer": "Adam", "weight_decay": 0., "ridge_mu": .1,
        "kappa": .25, "group_size": 8, "rl_calibration_draws": 4, "chunk_size": 128,
        "training_rng_seed_offset": 17041, "calibration_rng_seed_offset": 27041,
        "temperature_recipe": ".5 * population std(initial train eligible logits), clamp [.25,2]",
        "training_pool": "full training patients, per-disease eligible masks, fixed K macro means",
        "selection": "val mAP, includes epoch0, min improvement1e-6, patience15; fixed epoch10 diagnostic",
        "amp": False, "dropout": False, "dtype": "float32", "deterministic_mode": "strict",
        "score_normalization": "rank only: centered current train logits / current eligible std * fixed theta0 std; full derivatives",
        "normalization_std_epsilon": 1e-6,
        "reference_std_recipe": "population std(theta0 train eligible raw logits), no temperature factor or clamp"}
NORMALIZATION_CONFIG_KEYS = {"score_normalization", "normalization_std_epsilon", "reference_std_recipe"}
SOURCES = ("ranking_study/scale_invariant.py", "ranking_study/scale_invariant_trainer.py",
           "scripts/run_scale_invariant_ranking.py", "docs/group_ranking_scale_invariant_protocol.md")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def frozen_candidate():
    # The old verifier follows its explicit frozen lists, not a glob that would
    # retroactively add this new package's files to the old experiment.
    old = verify_v1()
    diagnostics = read_json(DIAGNOSTICS)
    require(diagnostics["source_manifest_sha256"] == file_sha(OLD_MANIFEST)
            and diagnostics["frozen_source_sha256"] == old["source_sha256"],
            "Triggering gradient diagnostics do not belong to the frozen v1 study")
    return {"schema_version": 1, "study": "single scale-invariant ranking refinement, frozen causal encoder",
            "specification": SPEC, "previous_manifest_path": relative(OLD_MANIFEST),
            "previous_manifest_sha256": file_sha(OLD_MANIFEST),
            "triggering_diagnostics_path": relative(DIAGNOSTICS), "triggering_diagnostics_sha256": file_sha(DIAGNOSTICS),
            "source_sha256": {name: file_sha(ROOT / name) for name in SOURCES},
            "feature_and_reference_sha256": old["feature_and_reference_sha256"],
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
            "feature_dir": relative(FEATURES / f"seed{seed}"),
            "output_dir": relative(STUDY / "runs" / (f"{method}-seed{seed}" + ("-repeat" if repeat else ""))),
            "wandb_mode": "offline", "wandb_project": "patient-timeline-forecast"}


def run_one(method, seed, device, repeat=False):
    manifest = verify()
    config = config_for(method, seed, device, repeat)
    if method != "bce":
        compare_v1_bce(seed)
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
    run = wandb.init(project=config["wandb_project"], name=f"scale-invariant-ranking-{tracking.name}",
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


def compare_v1_bce(seed):
    """The raw BCE arm must reproduce v1, not merely get a similar score."""
    import json
    import torch
    require(seed in SEEDS, "Undeclared BCE control seed")
    first, second = V1 / "runs" / f"bce-seed{seed}", STUDY / "runs" / f"bce-seed{seed}"
    evidence = {}
    for directory in (first, second):
        for name, fingerprint in read_json(directory / "artifact_sha256.json").items():
            require(file_sha(directory / name) == fingerprint, f"BCE control evidence changed: {directory / name}")
            evidence[relative(directory / name)] = fingerprint
    def rows(directory):
        return [{key: value for key, value in json.loads(line).items() if key != "elapsed_seconds"}
                for line in (directory / "history.jsonl").read_text().splitlines()]
    old_rows, new_rows = rows(first), rows(second)
    allowed_new_diagnostics = {"rank/normalization", "rank/raw_std_min", "rank/reference_std_min",
                               "rank/reference_std_max", "rank/scale_ratio_min", "rank/scale_ratio_max"}
    histories_equal = len(old_rows) == len(new_rows)
    for old_row, new_row in zip(old_rows, new_rows):
        histories_equal = histories_equal and all(key in new_row and new_row[key] == value for key, value in old_row.items())
        require((set(new_row) - set(old_row)).issubset(allowed_new_diagnostics),
                "Undeclared extra BCE-history fields would obscure the v1 control")
    configs = [{key: value for key, value in read_json(directory / "config.json").items()
                if key not in ({"methods", "output_dir"} | NORMALIZATION_CONFIG_KEYS)} for directory in (first, second)]
    initials = [read_json(directory / "initialization.json") for directory in (first, second)]
    allowed_initial_fields = {"reference_std", "reference_std_recipe", "normalization_reference_source"}
    require((set(initials[1]) - set(initials[0])).issubset(allowed_initial_fields), "Undeclared BCE initialization metadata")
    initial_equal = all(key in initials[1] and initials[1][key] == value for key, value in initials[0].items())
    def head(directory, name):
        return tensor_sha(torch.load(directory / name, map_location="cpu", weights_only=True))
    def full_model(directory):
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        return tensor_sha(checkpoint["model_state_dict"])
    summaries = [read_json(directory / "summary.json") for directory in (first, second)]
    comparison_keys = ("method", "seed", "best_epoch", "last_epoch", "no_update_selected",
                       "best_validation_mAP", "theta0_validation_mAP", "mAP_delta_from_theta0",
                       "epoch10_validation_mAP", "lambda", "frozen_backbone_export_verified", "test_metrics")
    checks = {"configuration_equal_except_declared_si_metadata": configs[0] == configs[1],
              "old_initialization_fields_equal": initial_equal,
              "environment_equal": read_json(first / "environment.json") == read_json(second / "environment.json"),
              "initial_head_equal": head(first, "initial_head.pt") == head(second, "initial_head.pt"),
              "all_old_history_non_time_fields_equal": histories_equal,
              "best_head_equal": head(first, "best_head.pt") == head(second, "best_head.pt"),
              "epoch10_head_equal": head(first, "epoch10_head.pt") == head(second, "epoch10_head.pt"),
              "full_model_tensors_equal": full_model(first) == full_model(second),
              "selection_summary_fields_equal": all(summaries[0][key] == summaries[1][key] for key in comparison_keys)}
    for name in ("val_predictions.csv", "test_predictions.csv", "train_predictions.csv", "epoch10_val_predictions.csv"):
        checks[f"{name}_equal"] = (first / name).read_bytes() == (second / name).read_bytes()
    result = {"passed": all(checks.values()), "seed": seed, "method": "bce",
              "checks": checks, "baseline_run_dir": relative(first), "current_run_dir": relative(second),
              "frozen_manifest_sha256": file_sha(STUDY / "frozen_manifest.json"),
              "artifact_sha256": evidence,
              "excluded_fields": {"config": sorted({"methods", "output_dir"} | NORMALIZATION_CONFIG_KEYS),
                                  "history": ["elapsed_seconds", *sorted(allowed_new_diagnostics)],
                                  "new_initialization": sorted(allowed_initial_fields)},
              "scope": "Exact raw-BCE same-code control. New rank-only normalization must not affect this arm."}
    path = STUDY / f"bce_v1_equivalence_seed{seed}.json"
    if path.exists():
        require(read_json(path) == result, "Refusing to replace different BCE equivalence evidence")
    else:
        write_json(path, result)
    require(result["passed"], "BCE no longer reproduces v1; stop before ranking attribution")
    return result


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
    def full_model(path):
        return tensor_sha(torch.load(path / "best.pt", map_location="cpu", weights_only=True)["model_state_dict"])
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
              "full_model_tensors_equal": full_model(first) == full_model(second),
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
    parser.add_argument("--stage", choices=("freeze", "verify", "train", "repeat", "bce-control"), required=True)
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
                if method == "bce":
                    print(compare_v1_bce(seed), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    elif args.stage == "bce-control":
        verify()
        for seed in SEEDS:
            print(compare_v1_bce(seed), flush=True)
    else:
        require(args.device == "cuda", "Declared A/A requires the same CUDA condition")
        verify()
        import torch
        for method in ("grpo_ap_si",):
            # Check the primary run/provenance before launching its repeat.
            require((STUDY / "runs" / f"{method}-seed17" / "summary.json").exists(), "Primary seed17 fit must complete first")
            run_one(method, 17, args.device)
            run_one(method, 17, args.device, repeat=True)
            print(compare_repeat(method), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
