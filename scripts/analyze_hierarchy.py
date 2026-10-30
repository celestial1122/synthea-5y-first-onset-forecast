"""Describe controlled single-model compression/hierarchy fits on validation only.

No seed probabilities are averaged and no model/seed is selected here. A partial
analysis includes a family only after every declared seed has completed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ehr_forecast.experiment_analysis import (
    UNCERTAINTY_NOTE, _load_runs, _validation_reference,
)
from ehr_forecast.hierarchy_analysis import paired_seed_group_comparisons
from ehr_forecast.metrics import evaluate_predictions


SEEDS = [17, 42, 73]
FAMILIES = ["raw_control", "date_recent", "date_full_flat", "date_full_hier", "date_full_overlap"]
REFINEMENT_FAMILIES = ["date_overlap16", "date_local2"]
NORMALIZATION_FAMILIES = ["date_sqrt_hier"]
LABELS = {
    "raw_control": "Raw-event control",
    "date_recent": "Same events, grouped by date",
    "date_full_flat": "More history, flat Transformer",
    "date_full_hier": "More history, hierarchical",
    "date_full_overlap": "Hierarchical + overlapping windows",
    "date_overlap16": "Finer 16-date overlapping windows",
    "date_local2": "16-date windows, 2 local / 1 global",
    "date_sqrt_hier": "16-date windows, sqrt-count scaling",
}
CONTRASTS = {
    "compression": ("raw_control", "date_recent", "Date compression on exactly the same recent raw events"),
    "coverage": ("date_recent", "date_full_flat", "Additional historical coverage under the same flat model"),
    "hierarchy": ("date_full_flat", "date_full_hier", "Hierarchical topology on identical date-unit inputs"),
    "overlap": ("date_full_hier", "date_full_overlap", "Overlapping local windows with unchanged parameters"),
    "end_to_end": ("raw_control", "date_full_overlap", "Full compression, coverage and hierarchy pipeline"),
}
REFINEMENT_CONTRASTS = {
    "window_granularity": ("date_full_overlap", "date_overlap16", "Window/stride 32/16 to 16/8 with the same 1-local/2-global allocation"),
    "layer_allocation": ("date_overlap16", "date_local2", "Local/global layer allocation 1+2 to 2+1 with the same 16/8 windows"),
}
NORMALIZATION_CONTRASTS = {
    "clinical_scale": ("date_local2", "date_sqrt_hier", "Date-bag mean to sum/sqrt(non-PAD count), keeping architecture, inputs and parameters fixed"),
}
REQUIRED_FILES = ["summary.json", "config.json", "metadata.json", "environment.json", "val_predictions.csv"]
CONTROL_KEYS = [
    "d_model", "n_heads", "n_layers", "model_max_events", "dropout", "batch_size", "epochs",
    "pretrain_epochs", "learning_rate", "weight_decay", "patience", "loss_positive_weight",
]
SECONDARY_FAMILIES = {
    "length-l512": "Earlier causal event Transformer with next-event pretraining",
    "no-pretrain-l512": "Earlier causal event Transformer without pretraining",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fixed_groups(path, ids):
    by_id = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "val":
                continue
            patient = row["patient_id"]
            if patient in by_id:
                raise ValueError("Duplicate validation patient in fixed history groups")
            count = int(row["full_event_count"])
            group = "long" if count > 255 else "short"
            if row["history_group_256"] != group:
                raise ValueError("Fixed history groups do not use the declared >255-event rule")
            by_id[patient] = group
    if set(by_id) != set(ids):
        raise ValueError("Fixed history groups must cover exactly the official validation population")
    groups = {name: np.array([by_id[patient] == name for patient in ids]) for name in ("long", "short")}
    # This release has a frozen, previously reported 365-patient validation cohort.
    if len(ids) == 365 and (groups["long"].sum() != 133 or groups["short"].sum() != 232):
        raise ValueError("The original 133-long/232-short validation definition changed")
    return groups


def _complete_families(study_root, expected_families=FAMILIES):
    completed, missing = {}, {}
    for family in expected_families:
        directories = []
        failures = []
        for seed in SEEDS:
            directory = study_root / "runs" / f"{family}-seed{seed}"
            absent = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
            if absent:
                failures.append({"seed": seed, "missing_files": absent})
                continue
            config = _read_json(directory / "config.json")
            if config.get("seed") != seed or config.get("family") != family:
                raise ValueError(f"Run directory does not match its declared family/seed: {directory}")
            directories.append(directory)
        if failures:
            missing[family] = failures
        else:
            completed[family] = directories
    return completed, missing


def _check_reference(prepared, reference):
    metadata = _read_json(prepared / "metadata.json")
    if metadata["target_codes"] != reference["target_codes"]:
        raise ValueError(f"Prepared targets differ: {prepared}")
    with np.load(prepared / "val.npz", allow_pickle=False) as arrays:
        ids = arrays["patient_ids"].astype(str).tolist()
        labels = arrays["labels"]
    if len(ids) != len(set(ids)) or set(ids) != set(reference["patient_ids"]):
        raise ValueError(f"Prepared validation patients differ: {prepared}")
    index = {patient: position for position, patient in enumerate(ids)}
    order = [index[patient] for patient in reference["patient_ids"]]
    if not np.array_equal(labels[order], reference["labels"]):
        raise ValueError(f"Prepared validation labels differ: {prepared}")


def _mean_sd(values):
    defined = [float(value) for value in values if value is not None]
    return (float(np.mean(defined)) if defined else None,
            float(np.std(defined, ddof=1)) if len(defined) > 1 else None)


def _family_summary(rows, masks):
    result = {}
    for group, mask in masks.items():
        metrics = [row[group] for row in rows]
        map_mean, map_std = _mean_sd([value["mAP"] for value in metrics])
        auc_mean, auc_std = _mean_sd([value["macro_auroc"] for value in metrics])
        brier_mean, brier_std = _mean_sd([value["brier"] for value in metrics])
        result[group] = {
            "mAP_mean": map_mean, "mAP_sample_std": map_std,
            "macro_auroc_mean": auc_mean, "macro_auroc_sample_std": auc_std,
            "brier_mean": brier_mean, "brier_sample_std": brier_std,
            "n_patients": int(mask.sum()),
            "n_positive_by_target": [entry["n_positive"] for entry in metrics[0]["per_condition"]],
            "n_auroc_valid": metrics[0]["n_auroc_valid"],
            "seeds": [row["seed"] for row in rows],
        }
    return result


def _secondary_references(reference, masks):
    """Describe prior single-model quality without treating it as a matched arm.

    The earlier causal architecture, classifier and position capacity differ
    from this study. These references prevent a weaker new control from hiding
    previously achieved quality, but their differences cannot identify the
    effect of compression/hierarchy. No additional bootstrap is calculated.
    """
    result = {}
    for family, description in SECONDARY_FAMILIES.items():
        directories = [ROOT / "artifacts/optimization/runs" / f"{family}-seed{seed}" for seed in SEEDS]
        missing = [{"seed": seed, "missing_files": [name for name in REQUIRED_FILES if not (directory / name).is_file()]}
                   for seed, directory in zip(SEEDS, directories)]
        missing = [entry for entry in missing if entry["missing_files"]]
        if missing:
            result[family] = {"complete": False, "missing_runs": missing, "description": description,
                              "secondary_descriptive_only": True, "architecture_matched_to_primary": False}
            continue
        runs = _load_runs(directories, reference, expected_seeds=SEEDS)
        for prepared_dir in {run["config"]["prepared_dir"] for run in runs}:
            _check_reference(ROOT / prepared_dir, reference)
        rows = []
        for run in runs:
            directory = Path(run["run_dir"])
            environment, summary = _read_json(directory / "environment.json"), _read_json(directory / "summary.json")
            metrics = {name: evaluate_predictions(reference["labels"][mask], run["probabilities"][mask],
                                                   reference["target_codes"], reference["target_descriptions"])
                       for name, mask in masks.items()}
            rows.append({"seed": run["seed"], "run_dir": str(directory), **metrics,
                         "best_epoch": summary["best_epoch"], "parameter_count": environment["parameter_count"],
                         "runtime_seconds": summary["elapsed_seconds"],
                         "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb")})
        result[family] = {
            "complete": True, "description": description, "per_seed": rows,
            "metrics_by_group": _family_summary(rows, masks), "seeds": SEEDS, "n_seeds": len(rows),
            "parameter_counts": [row["parameter_count"] for row in rows],
            "runtimes": [row["runtime_seconds"] for row in rows],
            "single_model_only": True, "secondary_descriptive_only": True,
            "architecture_matched_to_primary": False, "additional_bootstrap": False,
            "comparison_note": "Prior causal architecture/head/position capacity differs; score differences do not isolate a hierarchy effect. No probability ensemble or seed selection is used.",
        }
    return result


def _plots(study_root, families, comparisons, n_bootstrap, partial):
    if not families:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = study_root / "plots"
    root.mkdir(exist_ok=True)
    present = [name for name in [*FAMILIES, *REFINEMENT_FAMILIES, *NORMALIZATION_FAMILIES] if name in families]
    y = np.arange(len(present))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(3.4, .53 * len(present))))
    for axis, group in zip(axes, ("overall", "long")):
        means = [families[name]["metrics_by_group"][group]["mAP_mean"] for name in present]
        stds = [families[name]["metrics_by_group"][group]["mAP_sample_std"] for name in present]
        n = families[present[0]]["metrics_by_group"][group]["n_patients"]
        for index, family in enumerate(present):
            points = [row[group]["mAP"] for row in families[family]["per_seed"]]
            axis.scatter(points, np.full(len(points), index), color="#7B8691", s=20, alpha=.65,
                         label="Individual single-model fits" if index == 0 else None, zorder=2)
        axis.errorbar(means, y, xerr=stds, fmt="D", color="#163C64", capsize=3,
                      label="Single-model mean +/- sample SD", zorder=3)
        axis.set_yticks(y, [LABELS[name] for name in present])
        axis.invert_yaxis()
        axis.set(xlabel="Validation mAP over all target conditions (0-1)", ylabel="Experiment family",
                 title=f"{'All validation' if group == 'overall' else 'Fixed long-history'} patients (n={n})")
        axis.grid(axis="x", alpha=.2)
        axis.legend(fontsize=7, loc="best")
    fig.suptitle("Three fixed seeds: 17, 42, 73 | source: saved validation predictions | no probability ensembling"
                 + (" | INTERIM complete families only" if partial else ""), fontsize=9)
    fig.tight_layout()
    destination = root / "single_model_mAP.png"
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    outputs = [str(destination)]
    if comparisons:
        ordered = [name for name in [*CONTRASTS, *REFINEMENT_CONTRASTS, *NORMALIZATION_CONTRASTS] if name in comparisons]
        fig, axis = plt.subplots(figsize=(10.5, max(3.5, .6 * len(ordered))))
        for index, name in enumerate(ordered):
            interval = comparisons[name]["groups"]["long"]["delta_mAP"]
            axis.hlines(index, interval["ci_lower"], interval["ci_upper"], color="#163C64", linewidth=1.5)
            axis.scatter(interval["estimate"], index, color="#163C64", s=30)
        axis.axvline(0, color="#8C969F", linestyle="--", linewidth=1)
        axis.set_yticks(range(len(ordered)), [name.replace("_", " ") for name in ordered])
        axis.invert_yaxis()
        axis.set(xlabel="Mean paired SINGLE-MODEL mAP difference (candidate - baseline; 95% interval)",
                 ylabel="Controlled comparison", title="Fixed long-history validation patients: conditional patient-bootstrap intervals")
        axis.grid(axis="x", alpha=.2)
        fig.suptitle(f"Same patient draw across both families and all 3 seeds; {n_bootstrap} resamples | selection bias remains", fontsize=9)
        fig.tight_layout()
        destination = root / "long_history_paired_delta.png"
        fig.savefig(destination, dpi=170)
        plt.close(fig)
        outputs.append(str(destination))
    return outputs


def analyze(study_root: Path, groups_csv: Path, n_bootstrap=500, partial=False, include_refinements=False,
            include_normalization=False):
    study_root = study_root.resolve()
    if include_normalization and not include_refinements:
        raise ValueError("--include-normalization requires --include-refinements")
    expected_families = [*FAMILIES, *(REFINEMENT_FAMILIES if include_refinements else []),
                         *(NORMALIZATION_FAMILIES if include_normalization else [])]
    completed, missing = _complete_families(study_root, expected_families)
    if missing and not partial:
        detail = ", ".join(f"{family}: seeds {[item['seed'] for item in entries]}" for family, entries in missing.items())
        raise ValueError(f"Study incomplete ({detail}); use --partial for descriptive complete-family results only")
    reference = _validation_reference(study_root / "prepared_raw")
    groups = _fixed_groups(groups_csv, reference["patient_ids"])
    masks = {"overall": np.ones(len(reference["labels"]), dtype=bool), **groups}
    families, probabilities, control_signatures = {}, {}, []
    for family, directories in completed.items():
        runs = _load_runs(directories, reference, expected_seeds=SEEDS)
        _check_reference(ROOT / runs[0]["config"]["prepared_dir"], reference)
        rows = []
        for run in runs:
            directory = Path(run["run_dir"])
            environment, summary = _read_json(directory / "environment.json"), _read_json(directory / "summary.json")
            values = {name: evaluate_predictions(reference["labels"][mask], run["probabilities"][mask],
                                                  reference["target_codes"], reference["target_descriptions"])
                      for name, mask in masks.items()}
            rows.append({"seed": run["seed"], "run_dir": str(directory), **values,
                         "best_epoch": summary["best_epoch"], "parameter_count": environment["parameter_count"],
                         "runtime_seconds": summary["elapsed_seconds"],
                         "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb")})
            control_signatures.append({key: run["config"].get(key) for key in CONTROL_KEYS})
        families[family] = {"per_seed": rows, "metrics_by_group": _family_summary(rows, masks),
                            "parameter_counts": [row["parameter_count"] for row in rows],
                            "runtimes": [row["runtime_seconds"] for row in rows],
                            "seeds": SEEDS, "n_seeds": len(rows), "input_representation": _read_json(directories[0] / "metadata.json").get("representation"),
                            "single_model_only": True}
        probabilities[family] = {run["seed"]: run["probabilities"] for run in runs}
    controls_match = not control_signatures or all(value == control_signatures[0] for value in control_signatures)
    parameter_counts = {value for family in families.values() for value in family["parameter_counts"]}
    if not controls_match:
        raise ValueError("Declared architecture-independent training controls differ across families")
    if len(parameter_counts) > 1:
        raise ValueError("Parameter counts differ; investigate before labeling this study parameter-matched")
    comparisons = {}
    contrasts = {**CONTRASTS, **(REFINEMENT_CONTRASTS if include_refinements else {}),
                 **(NORMALIZATION_CONTRASTS if include_normalization else {})}
    for name, (baseline, candidate, description) in contrasts.items():
        if baseline not in probabilities or candidate not in probabilities:
            continue
        comparison = paired_seed_group_comparisons(reference["labels"], probabilities[baseline], probabilities[candidate],
                                                   reference["target_codes"], groups, reference["patient_ids"],
                                                   n_bootstrap=n_bootstrap, expected_seeds=SEEDS)
        comparisons[name] = {"baseline": baseline, "candidate": candidate, "description": description,
                             "overall": comparison["overall"], "groups": comparison["groups"],
                             "comparison_role": "adaptive_refinement" if name in {*REFINEMENT_CONTRASTS, *NORMALIZATION_CONTRASTS} else "core_controlled_contrast"}
    result = {
        "schema_version": 1, "complete": not missing, "partial": bool(missing), "split": "validation",
        "expected_families": expected_families, "expected_seeds": SEEDS, "families": families, "comparisons": comparisons,
        "core_study_complete": all(name in families for name in FAMILIES),
        "include_refinements": bool(include_refinements),
        "include_normalization": bool(include_normalization),
        "adaptive_refinements": {
            "included": bool(include_refinements),
            "families": [*(REFINEMENT_FAMILIES if include_refinements else []),
                          *(NORMALIZATION_FAMILIES if include_normalization else [])],
            "description": "Diagnostic-driven sensitivity analyses added after examining core-study behavior; not preregistered confirmatory evidence.",
            "rationale": "Coarse windows can leave too few global units, especially for short date sequences; evaluate finer windows and local/global depth allocation without changing parameters or training budget.",
            "normalization_sensitivity": {
                "included": bool(include_normalization),
                "comparison": "clinical_scale: date_local2 to date_sqrt_hier",
                "rationale": "Observed pooled-vector magnitude decreases with event count; sqrt-count scaling tests this mechanism without changing parameters. This association is not causal evidence.",
                "modality_specific_patch_used": False,
            },
            "selection_bias_note": "Validation has been used repeatedly, including to motivate these variants; paired intervals do not remove that adaptive selection bias.",
        },
        "missing_runs": missing, "n_patients": len(reference["patient_ids"]),
        "n_targets": len(reference["target_codes"]), "target_codes": reference["target_codes"],
        "group_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "long_group_rule": "Full deduplicated pre-anchor raw event count >255, fixed before this study",
        "epoch_selection": "Overall validation mAP for all models; subgroup metrics do not select epochs",
        "single_model_only": True, "uses_probability_ensemble": False,
        "selection": None, "selection_note": "This script reports all declared seeds and families without choosing a seed or model.",
        "training_controls_matched": controls_match, "parameter_counts_matched": len(parameter_counts) <= 1,
        "parameter_counts": sorted(parameter_counts), "n_bootstrap": n_bootstrap,
        "uncertainty_note": UNCERTAINTY_NOTE + " The estimand is mean paired single-model quality, conditional on these fixed seeds; seeds are not resampled.",
        "test_predictions_read": False,
    }
    result["secondary_references"] = _secondary_references(reference, masks)
    study_root.mkdir(parents=True, exist_ok=True)
    _write_json(study_root / "analysis.json", result)
    with (study_root / "single_model_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "seed", "group", "n_patients", "positive_instances", "mAP", "macro_auroc", "n_auroc_valid", "brier", "best_epoch", "parameter_count", "runtime_seconds", "peak_cuda_memory_mb"])
        writer.writeheader()
        for family, data in families.items():
            for row in data["per_seed"]:
                for group in masks:
                    metric = row[group]
                    writer.writerow({"family": family, "seed": row["seed"], "group": group,
                                     "n_patients": metric["n_patients"], "positive_instances": sum(item["n_positive"] for item in metric["per_condition"]),
                                     **{key: metric[key] for key in ("mAP", "macro_auroc", "n_auroc_valid", "brier")},
                                     **{key: row[key] for key in ("best_epoch", "parameter_count", "runtime_seconds", "peak_cuda_memory_mb")}})
    result["plots"] = _plots(study_root, families, comparisons, n_bootstrap, bool(missing))
    _write_json(study_root / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT / "artifacts/hierarchical")
    parser.add_argument("--groups-csv", type=Path, default=ROOT / "artifacts/optimization/history_patient_groups.csv")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--include-refinements", action="store_true", help="Also describe the two explicitly adaptive window/depth sensitivity analyses")
    parser.add_argument("--include-normalization", action="store_true", help="Also describe the final adaptive date-bag scaling sensitivity (requires --include-refinements)")
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be >=2")
    if args.include_normalization and not args.include_refinements:
        parser.error("--include-normalization requires --include-refinements")
    try:
        result = analyze(args.study_root, args.groups_csv, args.bootstrap, args.partial,
                         args.include_refinements, args.include_normalization)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"complete": result["complete"], "complete_families": list(result["families"]),
                      "missing_families": list(result["missing_runs"]), "single_model_only": True,
                      "scores": {name: data["metrics_by_group"] for name, data in result["families"].items()}},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
