"""Compare complete seed families on validation only; plot measured evidence."""

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
    average_validation_predictions, evaluate_groups, load_aligned_predictions,
    paired_bootstrap, summarize_runs, UNCERTAINTY_NOTE,
)
from ehr_forecast.metrics import evaluate_predictions
from ehr_forecast.train import write_json

SEEDS = [17, 42, 73]
OPT = ROOT / "artifacts/optimization"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def discover():
    grouped = {}
    for directory in sorted((OPT / "runs").glob("*")):
        if not (directory / "summary.json").exists():
            continue
        config = read(directory / "config.json")
        family = config.get("family", "original")
        grouped.setdefault(family, []).append(directory)
    original = ROOT / "artifacts/runs/transformer-seed42"
    grouped.setdefault("original", []).append(original)
    return {family: paths for family, paths in grouped.items()
            if sorted(read(path / "config.json")["seed"] for path in paths) == SEEDS}


def make_plots(families, comparisons, diagnostics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_root = OPT / "plots"
    plot_root.mkdir(exist_ok=True)
    keys = [key for key in ("length-l256", "length-l512", "length-l1024") if key in families]
    if keys:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        x = np.array([int(key.split("l")[-1]) for key in keys])
        for index, (key, length) in enumerate(zip(keys, x)):
            ys = [row["metrics"]["mAP"] for row in families[key]["per_seed"]]
            axes[0].scatter([length] * len(ys), ys, s=22, alpha=.65, color="#6F7C89",
                            label="Individual seed fits" if index == 0 else None)
        axes[0].errorbar(x, [families[key]["across_seeds"]["mAP"]["mean"] for key in keys],
                         yerr=[families[key]["across_seeds"]["mAP"]["sample_std"] for key in keys],
                         color="#163C64", capsize=4, marker="o", label="Mean +/- sample SD (3 seeds)")
        axes[0].set(xlabel="Maximum sequence length (tokens incl. BOS)", ylabel="Validation mAP (0-1)", title="Controlled length comparison")
        axes[0].legend(fontsize=7)
        for name in ("train", "val"):
            values = diagnostics["splits"][name]["lengths"]
            axes[1].plot(x, [100 * values[str(length)]["truncated_fraction"] for length in x], marker="o", label=name)
        axes[1].set(xlabel="Maximum sequence length (tokens incl. BOS)", ylabel="Patients truncated (%)", title="History coverage")
        axes[1].legend()
        for group in ("short", "long"):
            count = families[keys[0]]["equal_weight_ensemble"]["groups"][group]["n_patients"]
            axes[2].plot(x, [families[key]["equal_weight_ensemble"]["groups"][group]["metrics"]["mAP"] for key in keys], marker="o", label=f"{group} history (n={count})")
        axes[2].set(xlabel="Maximum sequence length (tokens incl. BOS)", ylabel="Validation mAP (0-1)", title="Within-group model comparison")
        axes[2].legend(fontsize=8)
        for axis in axes:
            axis.grid(alpha=.2)
        fig.suptitle("Source: official patient splits and saved runs | same encoder and initialization capacity", fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_root / "length_ablation.png", dpi=160)
        plt.close(fig)
    order = [key for key in ("original", "length-l256", "length-l512", "length-l1024", "summary-linear", "summary-mlp") if key in families]
    order.extend(key for key in families if key not in order)
    if order:
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4.2, .45 * len(order))))
        y = np.arange(len(order))
        for axis, metric in zip(axes, ("mAP", "macro_auroc")):
            means = [families[key]["across_seeds"][metric]["mean"] for key in order]
            stds = [families[key]["across_seeds"][metric]["sample_std"] for key in order]
            ensemble = [families[key]["equal_weight_ensemble"]["metrics"][metric] for key in order]
            axis.errorbar(means, y, xerr=stds, fmt="o", color="#163C64", capsize=3, label="Seed mean +/- sample SD")
            axis.scatter(ensemble, y, marker="x", color="#B85B25", label="Equal-weight 3-seed ensemble")
            axis.set_yticks(y, order)
            axis.invert_yaxis()
            axis.set(xlabel=f"Validation {metric} (0-1)", ylabel="Experiment family", title=f"{metric}: stability and ensembling")
            axis.grid(axis="x", alpha=.2)
            axis.legend(fontsize=7)
        fig.suptitle("Official validation set (365 patients, 40 targets) | epochs selected on this same validation set", fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_root / "family_comparison.png", dpi=160)
        plt.close(fig)
    if comparisons:
        selected = [(key, value["overall"]["delta_mAP"]) for key, value in comparisons.items()]
        fig, axis = plt.subplots(figsize=(10, max(3.2, .55 * len(selected))))
        for index, (name, value) in enumerate(selected):
            estimate = value["estimate"]
            axis.hlines(index, value["ci_lower"], value["ci_upper"], color="#163C64", linewidth=1.5)
            axis.plot(estimate, index, "o", color="#163C64")
        axis.set_yticks(range(len(selected)), [name.replace("__vs__", " minus ") for name, _ in selected])
        axis.invert_yaxis()
        axis.axvline(0, color="#888888", linestyle="--", linewidth=1)
        axis.set(xlabel="Difference in validation mAP (B - A; 95% patient bootstrap interval)", ylabel="Paired ensemble comparison", title="Conditional uncertainty on the selected validation models")
        axis.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(plot_root / "paired_intervals.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    groups = {"short": [], "long": []}
    with (OPT / "history_patient_groups.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "val":
                groups[row["history_group_256"]].append(row["patient_id"])
    with np.load(OPT / "prepared_l256/val.npz", allow_pickle=False) as arrays:
        ids, labels = arrays["patient_ids"].astype(str).tolist(), arrays["labels"].copy()
    metadata = read(OPT / "prepared_l256/metadata.json")
    codes = metadata["target_codes"]
    families, probabilities = {}, {}
    for family, directories in discover().items():
        config = read(directories[0] / "config.json")
        prepared = ROOT / config["prepared_dir"]
        with np.load(prepared / "val.npz", allow_pickle=False) as arrays:
            assert arrays["patient_ids"].astype(str).tolist() == ids
            np.testing.assert_array_equal(arrays["labels"], labels)
        result = summarize_runs(directories, prepared, groups, family_name=family, expected_seeds=SEEDS)
        result["runtime_seconds"] = [read(directory / "summary.json")["elapsed_seconds"] for directory in directories]
        families[family] = result
        probabilities[family] = average_validation_predictions(directories, prepared, expected_seeds=SEEDS)
    # Predeclared uniform cross-representation blend: best length + full-history MLP.
    length_families = [family for family in families if family.startswith("length-")]
    best_length = max(length_families, key=lambda key: families[key]["across_seeds"]["mAP"]["mean"]) if length_families else None
    sequence_families = [key for key in families if key.startswith(("length-", "pooling-", "no-pretrain-", "weighted-"))]
    summary_families = [key for key in ("summary-mlp", "summary-clip5") if key in families]
    best_sequence_mean = max(sequence_families, key=lambda key: families[key]["across_seeds"]["mAP"]["mean"]) if sequence_families else None
    best_summary_mean = max(summary_families, key=lambda key: families[key]["across_seeds"]["mAP"]["mean"]) if summary_families else None
    best_sequence = max(sequence_families, key=lambda key: families[key]["equal_weight_ensemble"]["metrics"]["mAP"]) if sequence_families else None
    best_summary = max(summary_families, key=lambda key: families[key]["equal_weight_ensemble"]["metrics"]["mAP"]) if summary_families else None
    blend = None
    if best_sequence and best_summary:
        blend_scores = (probabilities[best_sequence] + probabilities[best_summary]) / 2
        blend = {"families": [best_sequence, best_summary], "weights": [0.5, 0.5],
                 "metrics": evaluate_predictions(labels, blend_scores, codes, metadata["target_descriptions"]),
                 "groups": evaluate_groups(labels, blend_scores, codes, groups, ids)["groups"]}
        probabilities["uniform-sequence-summary-blend"] = blend_scores
    candidates = [*families, *( ["uniform-sequence-summary-blend"] if blend else [])]
    selected = max(candidates, key=lambda key: evaluate_predictions(labels, probabilities[key], codes)["mAP"]) if candidates else None
    selected_recipe = None
    if selected:
        member_families = blend["families"] if selected == "uniform-sequence-summary-blend" else [selected]
        selected_recipe = {"name": selected, "families": member_families, "n_members": 3 * len(member_families),
                           "weights": "equal probability weight for every member",
                           "metrics": evaluate_predictions(labels, probabilities[selected], codes, metadata["target_descriptions"]),
                           "selection_policy": "Compare every complete three-seed family ensemble and a fixed 50:50 best-sequence/best-summary blend; never select individual seeds",
                           "adaptive_selection": True,
                           "selection_history": "Initially screened single-model family means; revised after results because the deployed predictor is an ensemble. This adds validation selection bias and is disclosed."}
        probabilities["selected"] = probabilities[selected]
    comparisons = {}
    pairs = [("length-l256", name) for name in ("length-l512", "length-l1024")]
    if best_length:
        pairs.extend((best_length, name) for name in families if name.startswith("no-pretrain") or name.startswith("pooling"))
        pairs.extend([(best_length, "summary-mlp"), (best_length, "summary-linear")])
        pairs.extend(("pooling-l512", name) for name in families if name.startswith("weighted-"))
    if "summary-clip5" in families:
        pairs.append(("summary-mlp", "summary-clip5"))
    if "original" in probabilities:
        pairs.extend(("original", name) for name in ("summary-mlp", "uniform-sequence-summary-blend"))
        if selected:
            pairs.append(("original", "selected"))
    for a, b in pairs:
        if a not in probabilities or b not in probabilities:
            continue
        key = f"{b}__vs__{a}"
        comparisons[key] = {"baseline": a, "candidate": b,
                            "overall": paired_bootstrap(labels, probabilities[a], probabilities[b], codes, args.bootstrap)}
        if (a == "length-l256" and b.startswith("length-")) or b == "selected":
            for group, members in groups.items():
                mask = np.isin(ids, members)
                comparisons[key][group] = paired_bootstrap(labels[mask], probabilities[a][mask], probabilities[b][mask], codes, args.bootstrap)
    result = {"seeds": SEEDS, "groups": {key: len(value) for key, value in groups.items()},
              "families": families, "best_length_by_mean_seed_mAP": best_length,
              "best_sequence_by_mean_seed_mAP": best_sequence_mean, "best_summary_by_mean_seed_mAP": best_summary_mean,
              "best_sequence_by_ensemble_mAP": best_sequence, "best_summary_by_ensemble_mAP": best_summary,
              "selected_recipe": selected_recipe,
              "uniform_cross_representation_blend": blend, "paired_comparisons": comparisons,
              "uncertainty_note": UNCERTAINTY_NOTE,
              "all_controlled_lengths_complete": len(length_families) == 3,
              "selection_uses_test": False}
    if selected and "original" in families:
        prostate_codes = ["126906006", "92691004", "314994000"]
        before = {row["code"]: row for row in families["original"]["equal_weight_ensemble"]["metrics"]["per_condition"]}
        after = {row["code"]: row for row in selected_recipe["metrics"]["per_condition"]}
        contribution = sum(after[code]["average_precision"] - before[code]["average_precision"] for code in prostate_codes) / len(codes)
        net_gain = selected_recipe["metrics"]["mAP"] - families["original"]["equal_weight_ensemble"]["metrics"]["mAP"]
        result["gain_concentration"] = {
            "codes": prostate_codes, "net_mAP_gain_fraction": contribution / net_gain if net_gain else None,
            "unique_positive_validation_patients": int((labels[:, [codes.index(code) for code in prostate_codes]].sum(axis=1) > 0).sum()),
            "interpretation": "These related targets share positive patients; the gain is not three independent replications."
        }
    write_json(OPT / "analysis.json", result)
    with (OPT / "family_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "seed", "mAP", "macro_auroc", "short_mAP", "long_mAP"])
        for family, data in families.items():
            for run in data["per_seed"]:
                writer.writerow([family, run["seed"], run["metrics"]["mAP"], run["metrics"]["macro_auroc"], run["groups"]["short"]["metrics"]["mAP"], run["groups"]["long"]["metrics"]["mAP"]])
    make_plots(families, comparisons, read(OPT / "history_diagnostics.json"))
    print(json.dumps({family: {"mAP": data["across_seeds"]["mAP"], "macro_auroc": data["across_seeds"]["macro_auroc"],
                               "ensemble_mAP": data["equal_weight_ensemble"]["metrics"]["mAP"]} for family, data in families.items()}, indent=2))


if __name__ == "__main__":
    main()
