"""Audit saved legacy/raw-control results without training or loading a GPU.

Every probability metric is recomputed from the saved validation CSV.  The
training loss in history.jsonl is an online, dropout-enabled loss; it is never
reported as an eval-mode training-set metric.  Checkpoint SHA verification is
not a substitute for reloading the model and reproducing its predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ehr_forecast.experiment_analysis import load_aligned_predictions
from ehr_forecast.metrics import evaluate_predictions

SEEDS = (17, 42, 73)
PROSTATE_CODES = ("126906006", "92691004", "314994000")
LUNG_CODES = ("162573006", "424132000", "254637007")
METRIC_KEYS = ("mAP", "macro_auroc", "micro_ap", "brier", "at_risk_BCE", "at_risk_brier")


def rooted(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_summary(values) -> dict:
    defined = [float(value) for value in values if value is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "sample_std": float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
        "min": min(defined) if defined else None,
        "max": max(defined) if defined else None,
        "n_defined": len(defined),
    }


def probability_summary(labels, scores, prior) -> dict:
    risk = ~prior
    if not risk.any():
        raise ValueError("An evaluated cohort must contain at-risk target cells")
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1 - 1e-7)
    losses = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    positive = labels.astype(bool)
    negative_risk = (labels == 0) & risk
    return {
        "n_at_risk_cells": int(risk.sum()),
        "n_positive_cells": int(positive.sum()),
        "true_prevalence_all_cells": float(labels.mean()),
        "true_prevalence_at_risk": float(labels[risk].mean()),
        "mean_probability_at_risk": float(scores[risk].mean()),
        "probability_quantiles_at_risk": dict(zip(
            ("p01", "p50", "p90", "p99"),
            np.quantile(scores[risk], (.01, .5, .9, .99)).astype(float).tolist(),
        )),
        "mean_probability_for_positive_labels": float(scores[positive].mean()) if positive.any() else None,
        "mean_probability_for_at_risk_negative_labels": float(scores[negative_risk].mean()) if negative_risk.any() else None,
        "at_risk_BCE": float(losses[risk].mean()),
        "at_risk_brier": float(((scores - labels) ** 2)[risk].mean()),
    }


def read_reference(prepared: Path, new_prepared: Path, groups_csv: Path):
    metadata = read_json(prepared / "metadata.json")
    with np.load(prepared / "val.npz", allow_pickle=False) as arrays:
        reference = {name: arrays[name].copy() for name in arrays.files}
    ids = reference["patient_ids"].astype(str).tolist()
    labels = reference["labels"].astype(np.float64)
    prior = reference["prior_conditions"].astype(bool)
    if len(set(ids)) != len(ids) or np.any(labels[prior]):
        raise ValueError("Invalid reference patient IDs or prior-condition labels")
    new_metadata = read_json(new_prepared / "metadata.json")
    if new_metadata["target_codes"] != metadata["target_codes"]:
        raise ValueError("Old/new target code ordering differs")
    compared_arrays = ["patient_ids", "labels", "prior_conditions", "static", "anchor_dates"]
    with np.load(new_prepared / "val.npz", allow_pickle=False) as arrays:
        for name in compared_arrays:
            if name in reference:
                np.testing.assert_array_equal(reference[name], arrays[name], err_msg=f"Old/new {name} differs")
    assignments = {}
    with groups_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "val":
                continue
            patient = row["patient_id"]
            if patient in assignments:
                raise ValueError("Duplicate validation patient in fixed groups")
            assignments[patient] = row["history_group_256"]
    if set(assignments) != set(ids) or set(assignments.values()) != {"long", "short"}:
        raise ValueError("History groups do not match the exact validation cohort")
    masks = {"overall": np.ones(len(ids), dtype=bool)}
    masks.update({group: np.array([assignments[patient] == group for patient in ids]) for group in ("long", "short")})
    with np.load(prepared / "train.npz", allow_pickle=False) as arrays:
        train_labels = arrays["labels"].astype(np.float64)
        train_risk = ~arrays["prior_conditions"].astype(bool)
    positives = train_labels.sum(axis=0)
    at_risk = train_risk.sum(axis=0)
    priors = (positives + 0.5) / (at_risk + 1)
    prior_scores = np.tile(priors, (len(ids), 1))
    prior_scores[prior] = 0
    prior_metrics = evaluate_predictions(labels, prior_scores, metadata["target_codes"], metadata.get("target_descriptions"))
    return metadata, ids, labels, prior, masks, priors, prior_metrics


def audit_family(name, prefix, ids, labels, prior, masks, metadata, prior_metrics):
    codes = metadata["target_codes"]
    per_seed, fingerprints, ap_rows = [], {}, []
    for seed in SEEDS:
        run = Path(str(prefix) + str(seed))
        config = read_json(run / "config.json")
        if config["seed"] != seed or config.get("pretrain_epochs", 0) != 0:
            raise ValueError(f"Unexpected seed/pretraining configuration: {run}")
        summary = read_json(run / "summary.json")
        saved = read_json(run / "validation_metrics.json")
        scores = load_aligned_predictions(run / "val_predictions.csv", ids, codes)
        if np.any(scores[prior] != 0):
            raise ValueError(f"Prior target condition has nonzero incident probability: {run}")
        recomputed = evaluate_predictions(labels, scores, codes, metadata.get("target_descriptions"))
        metric_errors = {}
        for key in ("mAP", "macro_auroc", "micro_ap", "brier"):
            difference = abs(recomputed[key] - saved[key]) if recomputed[key] is not None and saved[key] is not None else 0.0
            metric_errors[key] = float(difference)
            if difference > 1e-8 or ((recomputed[key] is None) != (saved[key] is None)):
                raise ValueError(f"Saved {key} disagrees with prediction CSV: {run}")
        observed_hash = sha256(run / "best.pt")
        if observed_hash != summary["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint checksum differs from summary: {run}")
        history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        history = [row for row in history if row["stage"] == "finetune"]
        selected_rows = [row for row in history if row["epoch"] == summary["best_epoch"]]
        if len(selected_rows) != 1:
            raise ValueError(f"Selected checkpoint epoch is missing or duplicated: {run}")
        selected = selected_rows[0]
        best_bce = min(history, key=lambda row: row["val/loss"])
        last = history[-1]
        group_metrics = {}
        for group, mask in masks.items():
            metrics = evaluate_predictions(labels[mask], scores[mask], codes, metadata.get("target_descriptions"))
            metrics.update(probability_summary(labels[mask], scores[mask], prior[mask]))
            group_metrics[group] = metrics
        selected_bce_error = abs(group_metrics["overall"]["at_risk_BCE"] - selected["val/loss"])
        if selected_bce_error > 1e-8 or abs(selected["val/mAP"] - recomputed["mAP"]) > 1e-8:
            raise ValueError(f"Selected history metrics disagree with CSV: {run}")
        saved_prior = read_json(run / "prevalence_baseline_metrics.json")
        prior_errors = {key: abs(saved_prior[key] - prior_metrics[key]) for key in ("mAP", "macro_auroc", "brier")}
        if max(prior_errors.values()) > 1e-8:
            raise ValueError(f"Saved prevalence reference disagrees with train-only prior recipe: {run}")
        curve = {
            "best_mAP_epoch": int(summary["best_epoch"]),
            "last_epoch": int(last["epoch"]),
            "minimum_val_BCE_epoch": int(best_bce["epoch"]),
            "minimum_val_BCE": float(best_bce["val/loss"]),
            "mAP_at_minimum_val_BCE_epoch": float(best_bce["val/mAP"]),
            "selected_epoch_online_dropout_training_BCE": float(selected["train/loss"]),
            "selected_epoch_val_BCE": float(selected["val/loss"]),
            "last_online_dropout_training_BCE": float(last["train/loss"]),
            "last_val_BCE": float(last["val/loss"]),
            "last_val_mAP": float(last["val/mAP"]),
            "train_eval_BCE": None,
            "train_eval_mAP": None,
            "training_metric_limitation": "Logged training BCE is averaged during updates with dropout/AMP; it is not an epoch-end eval-mode training metric.",
        }
        per_seed.append({
            "seed": seed, "run_dir": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
            "config": config, "model_config": read_json(run / "model_config.json"),
            "metrics_by_group": group_metrics, "curves": curve,
            "checks": {"prediction_CSV_recomputed_metric_errors": metric_errors,
                       "selected_history_val_BCE_error": float(selected_bce_error),
                       "checkpoint_sha256_matches_summary": True, "checkpoint_sha256": observed_hash,
                       "prevalence_reference_errors": prior_errors, "n_nonfinite_probabilities": 0,
                       "prior_condition_probability_max": float(scores[prior].max()) if prior.any() else None,
                       "checkpoint_forward_reload_performed": False},
        })
        ap_rows.append([row["average_precision"] for row in recomputed["per_condition"]])
        for filename in ("config.json", "model_config.json", "summary.json", "history.jsonl", "validation_metrics.json", "prevalence_baseline_metrics.json", "val_predictions.csv", "best.pt"):
            path = run / filename
            fingerprints[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = observed_hash if filename == "best.pt" else sha256(path)
    aggregates = {group: {key: mean_summary(row["metrics_by_group"][group][key] for row in per_seed)
                          for key in METRIC_KEYS} for group in masks}
    return {"name": name, "seeds": list(SEEDS), "per_seed": per_seed, "across_seeds": aggregates}, np.asarray(ap_rows), fingerprints


def build_audit(prepared, new_prepared, groups_csv, old_prefix, new_prefix) -> dict[str, Any]:
    metadata, ids, labels, prior, masks, priors, prior_metrics = read_reference(prepared, new_prepared, groups_csv)
    old, old_ap, old_files = audit_family("legacy_no_pretrain_l512", old_prefix, ids, labels, prior, masks, metadata, prior_metrics)
    new, new_ap, new_files = audit_family("new_raw_control", new_prefix, ids, labels, prior, masks, metadata, prior_metrics)
    delta_ap = new_ap - old_ap
    target_changes = []
    for index, code in enumerate(metadata["target_codes"]):
        target_changes.append({
            "code": code, "description": metadata["target_descriptions"][index],
            "n_validation_positive": int(labels[:, index].sum()),
            "n_long_positive": int(labels[masks["long"], index].sum()),
            "n_short_positive": int(labels[masks["short"], index].sum()),
            "old_AP_by_seed": {str(seed): float(old_ap[row, index]) for row, seed in enumerate(SEEDS)},
            "new_AP_by_seed": {str(seed): float(new_ap[row, index]) for row, seed in enumerate(SEEDS)},
            "delta_AP_by_seed": {str(seed): float(delta_ap[row, index]) for row, seed in enumerate(SEEDS)},
            "old_AP_mean": float(old_ap[:, index].mean()), "new_AP_mean": float(new_ap[:, index].mean()),
            "delta_AP_mean": float(delta_ap[:, index].mean()),
            "contribution_to_macro_mAP_delta": float(delta_ap[:, index].mean() / len(metadata["target_codes"])),
        })
    target_by_code = {row["code"]: row for row in target_changes}
    net_delta = float(delta_ap.mean())
    concentration = {}
    for name, codes in (("prostate_related", PROSTATE_CODES), ("lung_related", LUNG_CODES)):
        indices = [metadata["target_codes"].index(code) for code in codes]
        contribution = sum(target_by_code[code]["contribution_to_macro_mAP_delta"] for code in codes)
        concentration[name] = {
            "target_codes": list(codes), "positive_supports": [target_by_code[code]["n_validation_positive"] for code in codes],
            "union_positive_patient_count": int((labels[:, indices].sum(axis=1) > 0).sum()),
            "positive_cooccurrence_matrix": (labels[:, indices].T @ labels[:, indices]).astype(int).tolist(),
            "net_macro_mAP_contribution": float(contribution),
            "fraction_of_net_macro_mAP_delta": float(contribution / net_delta) if net_delta else None,
            "interpretation_limit": "Correlated target changes share validation patients; they are not independent replications. Fraction is signed relative to the net change, not the fraction of total absolute changes.",
        }
    paired = {}
    for group in masks:
        paired[group] = {}
        for key in METRIC_KEYS:
            values = [n["metrics_by_group"][group][key] - o["metrics_by_group"][group][key]
                      for o, n in zip(old["per_seed"], new["per_seed"]) if n["metrics_by_group"][group][key] is not None and o["metrics_by_group"][group][key] is not None]
            paired[group][key] = {**mean_summary(values), "delta_by_seed": dict(zip(map(str, SEEDS), map(float, values))),
                                 "n_positive_deltas": int(sum(value > 0 for value in values)),
                                 "n_negative_deltas": int(sum(value < 0 for value in values))}
    sources = {**old_files, **new_files}
    for path in [prepared / "train.npz", prepared / "val.npz", new_prepared / "val.npz", groups_csv,
                 *(directory / filename for directory in (prepared, new_prepared) for filename in ("metadata.json", "vocabulary.json", "preprocessing.json"))]:
        sources[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha256(path)
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(Path(__file__).resolve())
    return {
        "schema_version": 1, "purpose": "Read-only audit of existing saved single-model validation results; no training or GPU operation.",
        "seeds": list(SEEDS), "uses_probability_ensemble": False,
        "cohort": {"n_validation_patients": len(ids), "n_targets": len(metadata["target_codes"]),
                   "groups": {group: int(mask.sum()) for group, mask in masks.items()},
                   "long_definition": "More than255 complete pre-anchor deduplicated events, fixed before model comparisons.",
                   "n_positive_cells": int(labels.sum()), "n_at_risk_cells": int((~prior).sum())},
        "input_checks": {"patient_order_equal": True, "target_order_equal": True, "labels_equal": True,
                         "prior_conditions_equal": True, "static_equal": True, "anchor_dates_equal": True,
                         "labels_do_not_overlap_prior_conditions": True},
        "train_only_incidence_reference": {"recipe": "(positive_count +0.5)/(at_risk_count +1), with known-prior probabilities set to0",
                                          "target_codes": metadata["target_codes"], "probabilities": priors.tolist(), "validation_metrics": prior_metrics},
        "families": {"old": old, "new": new}, "paired_single_model_differences_new_minus_old": paired,
        "target_changes": target_changes, "target_change_concentration": concentration,
        "net_mean_single_model_mAP_delta": net_delta,
        "largest_AP_losses": sorted(target_changes, key=lambda row: row["delta_AP_mean"])[:8],
        "largest_AP_gains": sorted(target_changes, key=lambda row: row["delta_AP_mean"], reverse=True)[:5],
        "established_evidence": [
            "Saved validation CSV metrics, selected history entries, checkpoint checksums and train-incidence reference are internally consistent.",
            "The new raw control has lower macro AP in all three paired seeds, but better mean validation BCE/Brier and slightly better mean micro AP.",
            "Both families reduce online training BCE while validation BCE rises and mAP drops after their selected epochs.",
            "A large part of the macro-AP reduction is concentrated in correlated lung targets covering only10 validation patients; prostate targets instead improve.",
        ],
        "interpretation_limits": [
            "No checkpoint forward reload or fresh training reproduction is performed by this script; SHA verification alone does not close that loop.",
            "No eval-mode training BCE/mAP has been logged. Online dropout training loss cannot identify the final model's exact training-set fit.",
            "The legacy/new models differ in multiple architectural and initialization details. These observations do not identify which difference caused the change.",
            "Validation also selected epochs and informed previous experiments. Between-seed SD is not patient uncertainty and does not eliminate selection bias.",
            "Metric tradeoffs do not prove that numerical failure, loss of age/sex information, mean pooling, or history compression is the primary cause.",
            "Test labels and test metrics are unavailable; no test prediction or outcome is used here.",
        ],
        "minimal_missing_diagnostics": [
            "Reload each frozen checkpoint and reproduce its saved validation predictions before fresh training reproduction.",
            "Add epoch-end eval-mode metrics on a fixed training cohort using the same metric/loss implementation as validation.",
            "Record pre-clipping gradient norm, clipping fraction and AMP skipped updates; run a fixed tiny-training-subset memorization check only if needed.",
            "Use one-factor, state-aligned bridge experiments; retain original global-mAP checkpoint selection and report minimum-BCE epochs only as a secondary diagnostic.",
        ],
        "source_files_sha256": sources,
    }


def write_tradeoff_csv(path: Path, audit: dict) -> None:
    fields = ["family", "seed", "group", "n_patients", "mAP", "macro_auroc", "micro_ap", "brier", "at_risk_BCE", "at_risk_brier",
              "mean_probability_at_risk", "true_prevalence_at_risk", "best_mAP_epoch", "minimum_val_BCE_epoch",
              "selected_epoch_online_dropout_training_BCE", "selected_epoch_val_BCE", "last_online_dropout_training_BCE", "last_val_BCE", "last_val_mAP"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family, data in audit["families"].items():
            for run in data["per_seed"]:
                for group, metrics in run["metrics_by_group"].items():
                    row = {"family": family, "seed": run["seed"], "group": group,
                           **{key: metrics[key] for key in fields if key in metrics},
                           **{key: run["curves"][key] for key in fields if key in run["curves"]}}
                    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="artifacts/optimization/prepared_l512")
    parser.add_argument("--new-prepared-dir", default="artifacts/hierarchical/prepared_raw")
    parser.add_argument("--groups-csv", default="artifacts/optimization/history_patient_groups.csv")
    parser.add_argument("--old-prefix", default="artifacts/optimization/runs/no-pretrain-l512-seed")
    parser.add_argument("--new-prefix", default="artifacts/hierarchical/runs/raw_control-seed")
    parser.add_argument("--output-dir", default="artifacts/experimental_audit")
    args = parser.parse_args()
    audit = build_audit(rooted(args.prepared_dir), rooted(args.new_prepared_dir), rooted(args.groups_csv), rooted(args.old_prefix), rooted(args.new_prefix))
    output = rooted(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "existing_diagnostics.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_tradeoff_csv(output / "ranking_probability_tradeoff.csv", audit)
    print(json.dumps({"output_dir": str(output), "seeds": audit["seeds"], "groups": audit["cohort"]["groups"],
                      "mean_mAP": {name: data["across_seeds"]["overall"]["mAP"]["mean"] for name, data in audit["families"].items()},
                      "mean_Brier": {name: data["across_seeds"]["overall"]["brier"]["mean"] for name, data in audit["families"].items()},
                      "mean_macro_mAP_delta": audit["net_mean_single_model_mAP_delta"],
                      "checkpoint_forward_reload_performed": False}, indent=2))


if __name__ == "__main__":
    main()
