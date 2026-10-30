"""Read-only checkpoint-selection, fixed-epoch and saved-score tie audit.

No model is fitted or loaded. Original runs and frozen runtime sources are read
only. Per-epoch condition scores cannot be reconstructed from aggregate logs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/experimental_audit_strict"
PREPARED = ROOT / "artifacts/optimization/prepared_l512"
OUTPUT = ROOT / "artifacts/ranking_diagnosis"
SEEDS = (17, 42, 73)
CELLS = ("C0R0", "C1R0", "C0R1", "C1R1")
FIXED_EPOCHS = (4, 6, 8, 10)
CONTRASTS = {
    "C_at_R0": ("C1R0", "C0R0"),
    "C_at_R1": ("C1R1", "C0R1"),
    "R_at_C0": ("C0R1", "C0R0"),
    "R_at_C1": ("C1R1", "C1R0"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
            "n_negative": int((values < 0).sum()), "n_positive": int((values > 0).sum())}


def run_dir(cell, seed):
    suffix = "-repeat-a" if cell == "C0R0" else ""
    return AUDIT / "runs" / f"{cell}-seed{seed}{suffix}"


def frozen_inputs():
    manifest = read(AUDIT / "frozen_manifest.json")
    for name, expected in manifest["source_sha256"].items():
        require(sha(ROOT / name) == expected, f"Frozen source changed: {name}")
    for name, expected in manifest["prepared_sha256"].items():
        require(sha(PREPARED / name) == expected, f"Frozen prepared input changed: {name}")
    for name, expected in manifest.get("auxiliary_inputs_sha256", {}).items():
        require(sha(ROOT / name) == expected, f"Frozen auxiliary input changed: {name}")
    audit = read(AUDIT / "audit_summary.json")
    require(audit["strict"] is True and audit["factorial_controls"]["passed"] is True,
            "Prior strict control verification did not pass")
    source = {str((AUDIT / "frozen_manifest.json").relative_to(ROOT)): sha(AUDIT / "frozen_manifest.json"),
              str((AUDIT / "audit_summary.json").relative_to(ROOT)): sha(AUDIT / "audit_summary.json")}
    return manifest, source


def read_scores(path, ids, codes):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader) == ["patient_id", *codes], f"Target order differs: {path}")
        rows = list(reader)
    require([row[0] for row in rows] == ids, f"Patient order differs: {path}")
    require(all(len(row) == len(codes) + 1 for row in rows), f"Wrong CSV shape: {path}")
    strings = [row[1:] for row in rows]
    scores = np.asarray(strings, dtype=np.float64)
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
            f"Invalid scores: {path}")
    # The writer uses .10g on float32 sigmoid output. Check recoverability,
    # rather than treating model/numerical ties as decimal-serialization ties.
    recovered = scores.astype(np.float32)
    exact_writer_roundtrip = all(text == format(float(value), ".10g")
                                 for row, values in zip(strings, recovered)
                                 for text, value in zip(row, values))
    preserved_unique_counts = all(np.unique(scores[:, j]).size == np.unique(recovered[:, j]).size
                                 for j in range(scores.shape[1]))
    return scores, {"format": ".10g", "all_entries_match_recovered_float32_writer": exact_writer_roundtrip,
                    "float32_roundtrip_preserves_unique_counts_per_target": preserved_unique_counts,
                    "scope": "Can recover a float32 matrix that exactly reserializes to these CSV strings. "
                             "No pre-CSV logits were saved; this does not identify the origin of numerical ties."}


def tie_ap_bounds(labels, scores):
    """Extremal rank AP over all refinements of exact-score tie groups."""
    truth = np.asarray(labels, dtype=bool)
    positives = int(truth.sum())
    if not positives:
        return 0., 0.
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_truth = scores[order], truth[order]
    cuts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores)]
    previous_n = previous_p = 0
    best_sum = worst_sum = 0.
    for start, end in zip(cuts[:-1], cuts[1:]):
        p = int(sorted_truth[start:end].sum())
        n = int(end - start - p)
        if p:
            sequence = np.arange(1, p + 1, dtype=np.float64)
            best_sum += float(((previous_p + sequence) / (previous_n + sequence)).sum())
            worst_sum += float(((previous_p + sequence) / (previous_n + n + sequence)).sum())
        previous_n += end - start
        previous_p += p
    return worst_sum / positives, best_sum / positives


def tie_stats(labels, scores, eligible):
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positive_count = np.bincount(inverse, weights=labels.astype(float), minlength=len(values))
    negative_count = counts - positive_count
    mixed = (positive_count > 0) & (negative_count > 0)
    eligible_values, eligible_counts = np.unique(scores[eligible], return_counts=True)
    del eligible_values
    ap = float(average_precision_score(labels, scores)) if labels.any() else 0.
    lower, upper = tie_ap_bounds(labels, scores)
    positive_tied = int(positive_count[counts > 1].sum())
    if not positive_tied:
        require(abs(lower - ap) < 1e-12 and abs(upper - ap) < 1e-12,
                "Without positive score ties, AP must be invariant to tie refinement")
    return {"ap": ap, "ap_min_over_exact_tie_refinements": lower,
            "ap_max_over_exact_tie_refinements": upper,
            "tie_refinement_ap_width": upper - lower,
            "n_unique_all": len(values), "n_score_tie_groups_all": int((counts > 1).sum()),
            "n_unique_at_risk": int(len(eligible_counts)),
            "n_at_risk_patients_in_ties": int(eligible_counts[eligible_counts > 1].sum()),
            "n_mixed_label_tie_groups": int(mixed.sum()),
            "n_positive_patients_in_any_ties": positive_tied,
            "n_positive_patients_in_mixed_ties": int(positive_count[mixed].sum()),
            "n_positive_negative_tied_pairs": int((positive_count * negative_count).sum())}


def main():
    _, sources = frozen_inputs()
    previous_audit = read(AUDIT / "audit_summary.json")
    metadata = read(PREPARED / "metadata.json")
    codes = metadata["target_codes"]
    descriptions = metadata["target_descriptions"]
    with np.load(PREPARED / "val.npz", allow_pickle=False) as arrays:
        ids = arrays["patient_ids"].astype(str).tolist()
        labels = arrays["labels"].astype(bool)
        prior = arrays["prior_conditions"].astype(bool)
    require(labels.shape == prior.shape == (len(ids), len(codes)), "Validation array shape differs")
    histories, selection, ties = {}, [], {}
    for cell in CELLS:
        for seed in SEEDS:
            folder = run_dir(cell, seed)
            for filename, expected in previous_audit["runs"][f"{cell}-seed{seed}"]["source_sha256"].items():
                require(sha(folder / filename) == expected, f"Audited run evidence changed: {folder / filename}")
            for filename in ("history.jsonl", "summary.json", "config.json", "val_predictions.csv", "validation_metrics.json"):
                sources[str((folder / filename).relative_to(ROOT))] = sha(folder / filename)
            config = read(folder / "config.json")
            require(config["seed"] == seed and config["patience"] == 8 and config["deterministic_mode"] == "strict",
                    f"Unexpected configuration: {folder}")
            rows = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
            require([row["epoch"] for row in rows] == list(range(1, len(rows) + 1)), "Invalid epoch sequence")
            histories[(cell, seed)] = rows
            summary = read(folder / "summary.json")
            best = rows[summary["best_epoch"] - 1]
            minimum = min(rows, key=lambda row: row["val/loss"])
            require(abs(max(row["val/mAP"] for row in rows) - best["val/mAP"]) < 1e-12,
                    "Saved epoch is not observed highest validation mAP")
            require(summary["best_validation_mAP"] == best["val/mAP"], "Summary score differs")
            record = {"cell": cell, "seed": seed, "stop_epoch": len(rows),
                      "mAP_best_epoch": best["epoch"], "BCE_min_epoch": minimum["epoch"],
                      "epoch_gap_mAP_minus_BCE": best["epoch"] - minimum["epoch"],
                      "at_mAP_best_mAP": best["val/mAP"], "at_mAP_best_BCE": best["val/loss"],
                      "at_BCE_min_mAP": minimum["val/mAP"], "at_BCE_min_BCE": minimum["val/loss"],
                      "mAP_gain_selecting_mAP_vs_BCE": best["val/mAP"] - minimum["val/mAP"],
                      "BCE_cost_selecting_mAP_vs_BCE": best["val/loss"] - minimum["val/loss"],
                      "stopped_best_plus_patience": len(rows) == best["epoch"] + config["patience"]}
            selection.append(record)
            scores, serialization = read_scores(folder / "val_predictions.csv", ids, codes)
            require(np.all(scores[prior] == 0), "Known-prior score was not zero")
            per_condition = [{"code": code, "description": descriptions[j], "n_positive": int(labels[:, j].sum()),
                              **tie_stats(labels[:, j], scores[:, j], ~prior[:, j])}
                             for j, code in enumerate(codes)]
            macro = float(np.mean([r["ap"] for r in per_condition]))
            require(abs(macro - best["val/mAP"]) < 1e-12, "Serialized mAP differs from selected metric")
            ties[f"{cell}-seed{seed}"] = {"serialization": serialization, "per_condition": per_condition,
                "mAP": macro,
                "mAP_min_over_exact_tie_refinements": float(np.mean([r["ap_min_over_exact_tie_refinements"] for r in per_condition])),
                "mAP_max_over_exact_tie_refinements": float(np.mean([r["ap_max_over_exact_tie_refinements"] for r in per_condition])),
                "positive_condition_cells_in_mixed_ties": int(sum(r["n_positive_patients_in_mixed_ties"] for r in per_condition)),
                "positive_condition_cells_in_any_ties": int(sum(r["n_positive_patients_in_any_ties"] for r in per_condition)),
                "n_at_risk_patients_in_ties_sum_over_targets": int(sum(r["n_at_risk_patients_in_ties"] for r in per_condition))}

    comparisons = {}
    for name, (new, old) in CONTRASTS.items():
        last = min(len(histories[(cell, seed)]) for cell in (new, old) for seed in SEEDS)
        curve = []
        for epoch in range(1, last + 1):
            map_deltas = [histories[(new, seed)][epoch - 1]["val/mAP"] - histories[(old, seed)][epoch - 1]["val/mAP"] for seed in SEEDS]
            bce_deltas = [histories[(new, seed)][epoch - 1]["val/loss"] - histories[(old, seed)][epoch - 1]["val/loss"] for seed in SEEDS]
            curve.append({"epoch": epoch, "mAP_delta": summarize(map_deltas), "BCE_delta": summarize(bce_deltas)})
        by_seed = {r["seed"]: r for r in selection if r["cell"] == new}
        old_by_seed = {r["seed"]: r for r in selection if r["cell"] == old}
        selected_delta = {metric: summarize([by_seed[seed][key] - old_by_seed[seed][key] for seed in SEEDS])
                          for metric, key in (("mAP", "at_mAP_best_mAP"), ("BCE", "at_mAP_best_BCE"))}
        min_bce_delta = [by_seed[seed]["at_BCE_min_BCE"] - old_by_seed[seed]["at_BCE_min_BCE"] for seed in SEEDS]
        selection_cost_delta = [by_seed[seed]["BCE_cost_selecting_mAP_vs_BCE"] - old_by_seed[seed]["BCE_cost_selecting_mAP_vs_BCE"] for seed in SEEDS]
        for seed, minimum_delta, cost_delta in zip(SEEDS, min_bce_delta, selection_cost_delta):
            require(abs(minimum_delta + cost_delta -
                        (by_seed[seed]["at_mAP_best_BCE"] - old_by_seed[seed]["at_mAP_best_BCE"])) < 1e-12,
                    "BCE selection accounting does not reconstruct selected score difference")
        tie_lower, tie_upper, tie_rows = [], [], []
        for seed in SEEDS:
            n, o = ties[f"{new}-seed{seed}"], ties[f"{old}-seed{seed}"]
            lower = n["mAP_min_over_exact_tie_refinements"] - o["mAP_max_over_exact_tie_refinements"]
            upper = n["mAP_max_over_exact_tie_refinements"] - o["mAP_min_over_exact_tie_refinements"]
            tie_lower.append(lower)
            tie_upper.append(upper)
            for nrow, orow in zip(n["per_condition"], o["per_condition"]):
                require(nrow["code"] == orow["code"], "Target order differs between tie reports")
                tie_rows.append({"seed": seed, "code": nrow["code"], "description": nrow["description"],
                                 "observed_AP_delta": nrow["ap"] - orow["ap"],
                                 "AP_delta_min_over_exact_tie_refinements": nrow["ap_min_over_exact_tie_refinements"] - orow["ap_max_over_exact_tie_refinements"],
                                 "AP_delta_max_over_exact_tie_refinements": nrow["ap_max_over_exact_tie_refinements"] - orow["ap_min_over_exact_tie_refinements"]})
        comparisons[name] = {"new": new, "old": old, "common_last_epoch_all_three_seeds": last,
            "selected_checkpoint_deltas": selected_delta,
            "BCE_selection_descriptive_identity": {
                "formula": "BCE(new,mAP-best)-BCE(old,mAP-best) = [min_BCE(new)-min_BCE(old)] + [(BCE_at_mAP_best-min_BCE)(new)-(BCE_at_mAP_best-min_BCE)(old)]",
                "observed_minimum_BCE_delta": summarize(min_bce_delta),
                "mAP_selection_BCE_cost_delta": summarize(selection_cost_delta),
                "interpretation": "An exact descriptive identity for observed trajectories, not a causal allocation or a replacement checkpoint-selection policy."},
            "fixed_epochs": [row for row in curve if row["epoch"] in FIXED_EPOCHS],
            "all_common_epoch_deltas": curve,
            "n_common_epochs_mean_BCE_improves": sum(row["BCE_delta"]["mean"] < 0 for row in curve),
            "n_common_epochs_mean_mAP_decreases": sum(row["mAP_delta"]["mean"] < 0 for row in curve),
            "tie_resolution_sensitivity": {"mean_mAP_delta_min": float(np.mean(tie_lower)),
                                           "mean_mAP_delta_max": float(np.mean(tie_upper)),
                                           "by_seed_min": dict(zip(map(str, SEEDS), tie_lower)),
                                           "by_seed_max": dict(zip(map(str, SEEDS), tie_upper)),
                                           "scope": "Adversarial extremal infinitesimal tie refinements of fixed saved scores; not retraining or a causal tie attribution. Grouped-threshold AP need not lie between refinement extrema in general; in this actual dataset no positive score ties occur, so AP is unchanged by all tie refinements.",
                                           "per_condition": tie_rows}}

    result = {"schema_version": 1, "seeds": list(SEEDS), "uses_probability_ensemble": False,
              "validation_patients": len(ids), "targets": len(codes),
              "fixed_epochs_declared_before_computation": list(FIXED_EPOCHS),
              "selection_by_run": selection, "comparisons": comparisons, "ties_by_run": ties,
              "source_sha256": sources,
              "limits": ["Aggregate histories contain val mAP/BCE but no per-epoch patient predictions or Brier.",
                         "Only mAP-selected best checkpoints/predictions are saved. Arbitrary per-epoch condition AP, Brier, patient ranks and calibration cannot be computed from these logs.",
                         "BCE-min epochs below are descriptive; no official checkpoint or reported prediction is replaced or selected anew.",
                         "Fixed-epoch comparisons can show that differing selection epochs are insufficient as the sole explanation; they cannot causally allocate the selected-checkpoint metric gap to selection versus representation.",
                         "Tie-refinement bounds hold fixed saved score groups and labels. They do not identify whether numerical precision during training/inference caused those groups.",
                         "All comparisons condition on three fitted seeds and the repeatedly used official validation patients; no statistical inference to future seeds or new patients is performed here."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "selection_tradeoff.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (OUTPUT / "selection_tradeoff.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection[0]))
        writer.writeheader()
        writer.writerows(selection)
    print(json.dumps({"outputs": ["selection_tradeoff.json", "selection_tradeoff.csv"],
                      "runs": len(selection), "comparisons": list(comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
