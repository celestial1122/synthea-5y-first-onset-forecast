"""Read-only, additive proper-score diagnosis of the frozen strict factorial.

No predictions are averaged: metrics and loss components are computed per seed.
The optional intervals condition on the three existing fitted checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ehr_forecast.metrics import evaluate_predictions

SEEDS = (17, 42, 73)
CELLS = ("C0R0", "C1R0", "C0R1", "C1R1")
CONTRASTS = {
    "attention_at_legacy": ("C0R0", "C1R0"),
    "attention_at_pooled": ("C0R1", "C1R1"),
    "readout_at_causal": ("C0R0", "C0R1"),
    "readout_at_bidirectional": ("C1R0", "C1R1"),
}
EPSILON = 1e-7


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolved(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_inputs():
    audit = ROOT / "artifacts/experimental_audit_strict"
    manifest = read(audit / "frozen_manifest.json")
    require(manifest["deterministic_mode"] == "strict" and manifest["seeds"] == list(SEEDS), "Wrong frozen study")
    for key in ("source_sha256", "auxiliary_inputs_sha256"):
        for path, expected in manifest[key].items():
            require(sha(ROOT / path) == expected, f"Frozen file changed: {path}")
    for key in ("original_frozen_manifest", "original_protocol", "strict_protocol"):
        require(sha(resolved(manifest[f"{key}_path"])) == manifest[f"{key}_sha256"], f"Frozen {key} changed")
    for name, expected in manifest["upstream_gates_sha256"].items():
        require(sha(ROOT / "artifacts/experimental_audit" / name) == expected, f"Upstream gate changed: {name}")
    prepared = resolved(manifest["prepared_dir"])
    for name, expected in manifest["prepared_sha256"].items():
        require(sha(prepared / name) == expected, f"Prepared input changed: {name}")
    summary = read(audit / "audit_summary.json")
    factorial = read(audit / "factorial_results.json")
    require(summary["factorial_controls"]["passed"] and factorial["controls_passed"], "Factorial controls did not pass")
    require(summary["strict"] and not summary["uses_probability_ensemble"], "Wrong audited analysis")
    require(summary["prepared_sha256"] == manifest["prepared_sha256"], "Summary inputs changed")
    meta = read(prepared / "metadata.json")
    codes = meta["target_codes"]
    with np.load(prepared / "val.npz", allow_pickle=False) as data:
        ids = data["patient_ids"].astype(str).tolist()
        y = data["labels"].astype(bool)
        prior = data["prior_conditions"].astype(bool)
    require(not np.any(y & prior), "Positive label overlaps prior condition")
    predictions, paths = {}, {}
    for cell in CELLS:
        predictions[cell] = {}
        for seed in SEEDS:
            run = summary["runs"][f"{cell}-seed{seed}"]
            folder = resolved(run["folder"])
            # The audited table stores hashes of all artifacts, not just CSVs.
            for name, expected in run["source_sha256"].items():
                require(sha(folder / name) == expected, f"Audited run artifact changed: {folder / name}")
            path = folder / "val_predictions.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            require(rows[0] == ["patient_id", *codes], f"Wrong target/header order: {path}")
            require([row[0] for row in rows[1:]] == ids, f"Wrong patient row order: {path}")
            p = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
            require(p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities")
            require(np.all(p[prior] == 0), "Prior predictions were not zeroed")
            recomputed = evaluate_predictions(y, p, codes)
            saved = read(folder / "validation_metrics.json")
            require(all(abs(recomputed[k] - saved[k]) <= 1e-8 for k in ("mAP", "brier", "micro_ap")), "Saved metric mismatch")
            selected_epoch = run["summary"]["best_epoch"]
            selected_row = next(row for row in run["history"] if row["epoch"] == selected_epoch)
            require(abs(float(bce(y, p)[~prior].mean()) - selected_row["val/loss"]) <= 1e-8,
                    "Selected-checkpoint at-risk BCE does not reproduce")
            predictions[cell][seed] = p
            paths[f"{cell}-seed{seed}"] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    with (ROOT / "artifacts/optimization/history_patient_groups.csv").open(encoding="utf-8-sig", newline="") as handle:
        assignments = {row["patient_id"]: row["history_group_256"] for row in csv.DictReader(handle) if row["split"] == "val"}
    require(set(assignments) == set(ids), "Subgroup rows differ")
    groups = {"overall": np.ones(len(ids), dtype=bool)}
    groups.update({name: np.asarray([assignments[p] == name for p in ids]) for name in ("long", "short")})
    require(groups["long"].sum() == 133 and groups["short"].sum() == 232, "Frozen subgroup sizes differ")
    provenance = {"frozen_manifest_sha256": sha(audit / "frozen_manifest.json"),
                  "audit_summary_sha256": sha(audit / "audit_summary.json"),
                  "factorial_results_sha256": sha(audit / "factorial_results.json"),
                  "script_sha256": sha(Path(__file__)), "predictions": paths}
    return meta, y, prior, predictions, groups, provenance


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0, "mean": None, "quantiles": None}
    return {"n": int(values.size), "mean": float(values.mean()),
            "standard_deviation": float(values.std()), "mean_square": float(np.mean(values ** 2)),
            "min": float(values.min()), "max": float(values.max()),
            "quantiles": {str(q): float(np.quantile(values, q)) for q in (.1, .25, .5, .75, .9, .95, .99)}}


def bce(y, p):
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return -(y * np.log(p) + (~y) * np.log1p(-p))


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def analyze_contrast(y, prior, old, new, codes, descriptions, name, group, bootstrap_weights):
    eligible = ~prior
    positive, negative = y, eligible & ~y
    n, t = y.shape
    balanced_valid = positive.any(axis=0) & negative.any(axis=0)
    denominator = {"brier": n * t, "at_risk_bce": int(eligible.sum())}
    per_seed, csv_rows, delta_matrices = [], [], []
    condition_rows = []
    for seed in SEEDS:
        a, z = old[seed], new[seed]
        da, dz = evaluate_predictions(y, a, codes), evaluate_predictions(y, z, codes)
        ba, bz = (a - y) ** 2, (z - y) ** 2
        ca, cz = bce(y, a) * eligible, bce(y, z) * eligible
        db, dc = bz - ba, (cz - ca) * eligible
        delta_matrices.append((db, dc))
        components = {}
        for metric, delta in (("brier", db), ("at_risk_bce", dc)):
            loss_first, loss_second = (ba, bz) if metric == "brier" else (ca, cz)
            components[metric] = {}
            for label, mask in (("positive", positive), ("at_risk_negative", negative), ("prior_negative", prior)):
                components[metric][label] = {"count": int(mask.sum()),
                    "contribution_to_global_delta": float(delta[mask].sum() / denominator[metric]),
                    "within_group_mean_loss_old": float(loss_first[mask].mean()) if mask.any() else None,
                    "within_group_mean_loss_new": float(loss_second[mask].mean()) if mask.any() else None,
                    "within_group_mean_loss_delta": float(delta[mask].mean()) if mask.any() else None}
            total = sum(value["contribution_to_global_delta"] for value in components[metric].values())
            expected = float(delta.sum() / denominator[metric])
            require(abs(total - expected) < 1e-13, f"Component reconstruction failed {name} {group} {seed} {metric}")
            components[metric]["total_delta"] = total
        require(abs(components["brier"]["total_delta"] - (dz["brier"] - da["brier"])) < 1e-12, "Brier reconstruction failed")
        aps_a = np.asarray([row["average_precision"] for row in da["per_condition"]])
        aps_z = np.asarray([row["average_precision"] for row in dz["per_condition"]])
        balanced = {}
        for metric, first_loss, second_loss in (("brier", ba, bz), ("bce", ca, cz)):
            per_label = {}
            for label, mask in (("positive", positive), ("at_risk_negative", negative)):
                counts = mask.sum(axis=0)[balanced_valid]
                per_label[label] = {"old": float(((first_loss * mask).sum(axis=0)[balanced_valid] / counts).mean()),
                                    "new": float(((second_loss * mask).sum(axis=0)[balanced_valid] / counts).mean())}
                per_label[label]["delta"] = per_label[label]["new"] - per_label[label]["old"]
            balanced[metric] = {"positive_class_macro_mean_loss": per_label["positive"],
                               "negative_class_macro_mean_loss": per_label["at_risk_negative"],
                               "old": .5 * sum(value["old"] for value in per_label.values()),
                               "new": .5 * sum(value["new"] for value in per_label.values()),
                               "delta": .5 * sum(value["delta"] for value in per_label.values())}
        row = {"seed": seed, "old": {"mAP": da["mAP"], "brier": da["brier"], "at_risk_bce": float(ca[eligible].mean())},
               "new": {"mAP": dz["mAP"], "brier": dz["brier"], "at_risk_bce": float(cz[eligible].mean())},
               "mAP_delta": dz["mAP"] - da["mAP"], "loss_components": components,
               "diagnostic_balanced_loss": balanced,
               "positive_geometric_mean_probability": {"old": float(np.exp(-ca[positive].mean())),
                                                       "new": float(np.exp(-cz[positive].mean()))},
               "probability_distributions": {label: {"old": distribution(a[mask]), "new": distribution(z[mask]),
                   "paired_probability_delta": distribution((z - a)[mask])}
                   for label, mask in (("positive", positive), ("at_risk_negative", negative), ("prior_negative", prior))}}
        per_seed.append(row)
        disease_sum = {"AP": 0., "brier": 0., "at_risk_bce": 0.}
        for j, code in enumerate(codes):
            col = {"contrast": name, "group": group, "seed": seed, "code": code, "description": descriptions[j],
                   "n_positive": int(positive[:, j].sum()), "n_at_risk_negative": int(negative[:, j].sum()),
                   "n_prior_negative": int(prior[:, j].sum()), "n_eligible": int(eligible[:, j].sum()),
                   "AP_old": float(aps_a[j]), "AP_new": float(aps_z[j]), "AP_delta": float(aps_z[j] - aps_a[j]),
                   "mAP_contribution": float((aps_z[j] - aps_a[j]) / t),
                   "brier_old": float(ba[:, j].mean()), "brier_new": float(bz[:, j].mean()),
                   "brier_delta": float(db[:, j].mean()), "brier_global_contribution": float(db[:, j].sum() / denominator["brier"]),
                   "bce_old": float(ca[:, j][eligible[:, j]].mean()) if eligible[:, j].any() else None,
                   "bce_new": float(cz[:, j][eligible[:, j]].mean()) if eligible[:, j].any() else None,
                   "bce_global_contribution": float(dc[:, j].sum() / denominator["at_risk_bce"])}
            for label, mask in (("positive", positive), ("at_risk_negative", negative), ("prior_negative", prior)):
                selected = mask[:, j]
                for metric, loss_old, loss_new, delta in (("brier", ba, bz, db), ("bce", ca, cz, dc)):
                    denom = denominator["brier" if metric == "brier" else "at_risk_bce"]
                    col[f"{label}_{metric}_global_contribution"] = float(delta[:, j][selected].sum() / denom)
                    col[f"{label}_{metric}_mean_delta"] = float(delta[:, j][selected].mean()) if selected.any() else None
                    col[f"{label}_{metric}_old_mean"] = float(loss_old[:, j][selected].mean()) if selected.any() else None
                    col[f"{label}_{metric}_new_mean"] = float(loss_new[:, j][selected].mean()) if selected.any() else None
                for label2, values in (("old", a), ("new", z), ("delta", z-a)):
                    col[f"{label}_p_{label2}_mean"] = float(values[:, j][selected].mean()) if selected.any() else None
            disease_sum["AP"] += col["mAP_contribution"]
            disease_sum["brier"] += col["brier_global_contribution"]
            disease_sum["at_risk_bce"] += col["bce_global_contribution"]
            csv_rows.append(col)
        require(abs(disease_sum["AP"] - row["mAP_delta"]) < 1e-12, "AP disease reconstruction failed")
        require(abs(disease_sum["brier"] - components["brier"]["total_delta"]) < 1e-12, "Brier disease reconstruction failed")
        require(abs(disease_sum["at_risk_bce"] - components["at_risk_bce"]["total_delta"]) < 1e-12, "BCE disease reconstruction failed")
    scalar_keys = [key for key in csv_rows[0] if key not in {"contrast", "group", "seed", "code", "description"}]
    for j, code in enumerate(codes):
        seed_rows = [row for row in csv_rows if row["code"] == code]
        condition_rows.append({"code": code, "description": descriptions[j],
            **{key: (float(np.mean([row[key] for row in seed_rows])) if seed_rows[0][key] is not None else None)
               for key in scalar_keys}})
    aggregate = {"mAP_delta": summarize([row["mAP_delta"] for row in per_seed]), "loss_components": {}}
    aggregate["diagnostic_balanced_loss"] = {metric: {key: summarize([row["diagnostic_balanced_loss"][metric][key] for row in per_seed])
        for key in ("old", "new", "delta")} for metric in ("brier", "bce")}
    for metric in ("brier", "at_risk_bce"):
        aggregate["loss_components"][metric] = {"total_delta": summarize([row["loss_components"][metric]["total_delta"] for row in per_seed])}
        for label in ("positive", "at_risk_negative", "prior_negative"):
            aggregate["loss_components"][metric][label] = {
                "count": int((positive if label == "positive" else negative if label == "at_risk_negative" else prior).sum()),
                "contribution_to_global_delta": summarize([row["loss_components"][metric][label]["contribution_to_global_delta"] for row in per_seed]),
                "within_group_mean_loss_delta": summarize([row["loss_components"][metric][label]["within_group_mean_loss_delta"] for row in per_seed])
                    if per_seed[0]["loss_components"][metric][label]["within_group_mean_loss_delta"] is not None else None,
                **{key: summarize([row["loss_components"][metric][label][key] for row in per_seed])
                   if per_seed[0]["loss_components"][metric][label][key] is not None else None
                   for key in ("within_group_mean_loss_old", "within_group_mean_loss_new")}}
    # Means of *loss differences*, not probability ensembles. All seeds share the denominator.
    mean_db = np.mean([pair[0] for pair in delta_matrices], axis=0)
    mean_dc = np.mean([pair[1] for pair in delta_matrices], axis=0)
    intervals = {}
    for metric, delta in (("brier", mean_db), ("at_risk_bce", mean_dc)):
        intervals[metric] = {}
        draws_denom = (n * t if metric == "brier" else bootstrap_weights @ eligible.sum(axis=1))
        for label, mask in (("positive", positive), ("at_risk_negative", negative), ("prior_negative", prior), ("total", np.ones_like(y))):
            draws = (bootstrap_weights @ (delta * mask).sum(axis=1)) / draws_denom
            intervals[metric][label] = {"estimate": float((delta * mask).sum() / denominator[metric]),
                "ci_lower": float(np.quantile(draws, .025)), "ci_upper": float(np.quantile(draws, .975))}
    by_ap_loss = sorted(condition_rows, key=lambda row: row["mAP_contribution"])
    by_brier_gain = sorted(condition_rows, key=lambda row: row["brier_global_contribution"])
    by_bce_gain = sorted(condition_rows, key=lambda row: row["bce_global_contribution"])
    report = {"old_cell": CONTRASTS[name][0], "new_cell": CONTRASTS[name][1], "n_patients": n,
              "n_positive_cells": int(positive.sum()), "n_at_risk_negative_cells": int(negative.sum()), "n_prior_negative_cells": int(prior.sum()),
              "per_seed": per_seed, "mean_single_model": aggregate, "conditional_patient_bootstrap": intervals,
              "diagnostic_balanced_loss_policy": {"definition": "Mean across valid classes of0.5*positive mean loss+0.5*at-risk-negative mean loss; diagnostic only, never training/checkpoint selection.",
                                                  "n_valid_targets": int(balanced_valid.sum()),
                                                  "valid_target_codes": [code for code, valid in zip(codes, balanced_valid) if valid]},
              "conditions": condition_rows, "largest_AP_losses": by_ap_loss[:10], "largest_AP_gains": by_ap_loss[-10:][::-1],
              "largest_Brier_gains": by_brier_gain[:10], "largest_BCE_gains": by_bce_gain[:10]}
    return report, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    require(args.bootstrap >= 100, "At least100 resamples required")
    meta, y, prior, predictions, groups, provenance = load_inputs()
    codes = meta["target_codes"]
    descriptions = meta["target_descriptions"]
    rng = np.random.default_rng(20261028)
    reports, rows = {}, []
    for group, mask in groups.items():
        counts = rng.multinomial(int(mask.sum()), np.full(int(mask.sum()), 1 / mask.sum()), size=args.bootstrap)
        reports[group] = {}
        for name, (a, z) in CONTRASTS.items():
            report, new_rows = analyze_contrast(y[mask], prior[mask], {s: predictions[a][s][mask] for s in SEEDS},
                {s: predictions[z][s][mask] for s in SEEDS}, codes, descriptions, name, group, counts)
            reports[group][name] = report
            rows.extend(new_rows)
    result = {"schema_version": 1, "scope": "Fixed official validation, mAP-selected strict checkpoints; no refitting or model changes.",
              "seeds": list(SEEDS), "uses_probability_ensemble": False, "bootstrap_resamples": args.bootstrap,
              "bootstrap_seed": 20261028, "confidence_level": .95,
              "bootstrap_scope": "Patient sampling conditional on the three fitted seeds and selected checkpoints; no multiplicity correction or correction for validation selection.",
              "definitions": {"Brier": "All patient-target cells; prior cells have y=p=0 and contribute0.",
                              "BCE": "Only at-risk cells, clip probabilities to[1e-7,1-1e-7].",
                              "mAP": "Mean AP over all40 targets; zero AP for no-positive targets.",
                              "positive_negative_components": "Each loss component uses the same global metric denominator and sums exactly to total delta. Also report within-component mean losses/deltas.",
                              "condition_Brier_weight": "Equal patients per disease implies equal disease weights, but positives are sparse within disease; do not say Brier simply gives common diseases more weight.",
                              "probability_distributions": "Computed per fitted seed, never on averaged probabilities.",
                              "difference_direction": "new minus old; AP higher better; BCE/Brier lower better."},
              "checks": {"source_data_artifact_hashes": True, "CSV_exact_order": True, "loss_and_disease_reconstruction": True},
              "provenance": provenance, "groups": reports}
    output = ROOT / "artifacts/ranking_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "score_components.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "condition_score_components.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"checks_passed": True, "conditions_rows": len(rows), "contrasts": list(CONTRASTS), "groups": list(groups), "bootstrap": args.bootstrap}))


if __name__ == "__main__":
    main()
