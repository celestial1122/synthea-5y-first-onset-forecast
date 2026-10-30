"""Read-only, tie-correct patient attribution of saved strict validation AP.

No training, checkpoint selection, probability ensembling or patient reordering.
Only the outputs below are written; all fitted evidence is hash checked.
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
COMPARISONS = {"C1R0_minus_C0R0": ("C0R0", "C1R0"),
               "C1R1_minus_C0R1": ("C0R1", "C1R1"),
               "C0R1_minus_C0R0": ("C0R0", "C0R1"),
               "C1R1_minus_C1R0": ("C1R0", "C1R1")}
READOUT_COMPARISONS = frozenset({"C0R1_minus_C0R0", "C1R1_minus_C1R0"})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_predictions(path, ids, codes):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    require(rows[0] == ["patient_id", *codes], f"Column order mismatch: {path}")
    require([r[0] for r in rows[1:]] == ids, f"Patient order mismatch: {path}")
    p = np.asarray([r[1:] for r in rows[1:]], dtype=np.float64)
    require(p.shape == (len(ids), len(codes)) and np.isfinite(p).all()
            and (p >= 0).all() and (p <= 1).all(), f"Bad probabilities: {path}")
    return p


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), f"Empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold_state(scores, truth, eligible, row):
    """Threshold precision reproduces sklearn's AP including tied scores."""
    p = scores[row]
    greater, equal = scores > p, scores == p
    positive = truth.astype(bool)
    negative = ~positive
    above, tied = int(greater.sum()), int(equal.sum())
    tp_ge = int(((greater | equal) & positive).sum())
    precision = tp_ge / (above + tied)
    return {
        "probability": float(p), "rank_best": above + 1, "rank_worst": above + tied,
        "at_risk_rank_best": int((greater & eligible).sum()) + 1,
        "at_risk_rank_worst": int(((greater | equal) & eligible).sum()),
        "positive_strictly_above": int((greater & positive).sum()),
        "positive_tied_including_self": int((equal & positive).sum()),
        "at_risk_negative_strictly_above": int((greater & negative & eligible).sum()),
        "at_risk_negative_tied": int((equal & negative & eligible).sum()),
        "prior_negative_strictly_above": int((greater & negative & ~eligible).sum()),
        "threshold_precision": float(precision),
    }


def aggregate_positive_rows(rows, n_targets):
    by_cell = {}
    for row in rows:
        key = (row["patient_id"], row["code"])
        by_cell.setdefault(key, []).append(row)
    cells = []
    for (patient, code), values in by_cell.items():
        require({r["seed"] for r in values} == set(SEEDS), "Incomplete positive cell seeds")
        base = values[0]
        summary = {k: base[k] for k in ("patient_id", "patient_alias", "code", "description",
                                       "n_positive", "history_group", "full_event_count",
                                       "retained_event_count", "dropped_event_count",
                                       "age_at_anchor_derived_years", "sex")}
        summary.update({"mean_mAP_contribution": float(np.mean([r["mAP_contribution_delta"] for r in values])),
                        "mean_AP_contribution": float(np.mean([r["AP_contribution_delta"] for r in values])),
                        "mean_probability_before": float(np.mean([r["probability_before"] for r in values])),
                        "mean_probability_after": float(np.mean([r["probability_after"] for r in values])),
                        "mean_rank_worst_before": float(np.mean([r["rank_worst_before"] for r in values])),
                        "mean_rank_worst_after": float(np.mean([r["rank_worst_after"] for r in values])),
                        "negative_contribution_seeds": sum(r["mAP_contribution_delta"] < 0 for r in values),
                        "probability_fell_seeds": sum(r["probability_delta"] < 0 for r in values),
                        "rank_worsened_seeds": sum(r["rank_worst_delta"] > 0 for r in values),
                        "per_seed": values})
        cells.append(summary)
    cells.sort(key=lambda r: r["mean_mAP_contribution"])
    by_patient = {}
    for row in cells:
        key = row["patient_id"]
        if key not in by_patient:
            by_patient[key] = {k: row[k] for k in ("patient_id", "patient_alias", "history_group",
                              "full_event_count", "age_at_anchor_derived_years", "sex")}
            by_patient[key].update(mean_mAP_contribution=0.0, condition_cells=[])
        by_patient[key]["mean_mAP_contribution"] += row["mean_mAP_contribution"]
        by_patient[key]["condition_cells"].append({k: row[k] for k in ("code", "description", "mean_mAP_contribution")})
    patients = sorted(by_patient.values(), key=lambda r: r["mean_mAP_contribution"])
    return cells, patients


def main():
    manifest = read(AUDIT / "frozen_manifest.json")
    for name, expected in manifest["source_sha256"].items():
        require(sha(ROOT / name) == expected, f"Frozen source changed: {name}")
    for name, expected in manifest["prepared_sha256"].items():
        require(sha(PREPARED / name) == expected, f"Prepared data changed: {name}")
    for name, expected in manifest["auxiliary_inputs_sha256"].items():
        require(sha(ROOT / name) == expected, f"Auxiliary data changed: {name}")
    require(sha(ROOT / manifest["strict_protocol_path"]) == manifest["strict_protocol_sha256"], "Strict protocol changed")
    summary = read(AUDIT / "audit_summary.json")
    require(summary["strict"] and summary["factorial_controls"]["passed"]
            and not summary["uses_probability_ensemble"], "Strict audit controls did not pass")
    metadata = read(PREPARED / "metadata.json")
    preprocessing = read(PREPARED / "preprocessing.json")
    codes = metadata["target_codes"]
    descriptions = dict(zip(codes, metadata["target_descriptions"]))
    with np.load(PREPARED / "val.npz", allow_pickle=False) as a:
        data = {k: a[k].copy() for k in a.files}
    ids = data["patient_ids"].astype(str).tolist()
    labels, prior = data["labels"].astype(bool), data["prior_conditions"].astype(bool)
    n, n_targets = labels.shape
    require((n, n_targets) == (365, 40) and not (labels & prior).any(), "Wrong cohort/incident labels")
    with (ROOT / "artifacts/optimization/history_patient_groups.csv").open(newline="", encoding="utf-8") as f:
        group_rows = {r["patient_id"]: r for r in csv.DictReader(f) if r["split"] == "val"}
    require(set(group_rows) == set(ids), "Patient groups incomplete")
    aliases = {p: f"V{i + 1:03d}" for i, p in enumerate(ids)}
    context = {}
    for i, patient in enumerate(ids):
        static = data["static"][i]
        count = int(data["full_history_length"][i])
        require(count == int(group_rows[patient]["full_event_count"]), "History counts disagree")
        age = None if static[4] else float(static[0] * preprocessing["static"]["age_scale"] + preprocessing["static"]["age_mean"])
        context[patient] = dict(patient_alias=aliases[patient], history_group=group_rows[patient]["history_group_256"],
                                full_event_count=count, retained_event_count=min(count, 511),
                                dropped_event_count=max(count - 511, 0), age_at_anchor_derived_years=age,
                                sex=["female", "male", "unknown"][int(np.argmax(static[1:4]))])
    predictions, evidence_sha = {}, {}
    for cell in ("C0R0", "C1R0", "C0R1", "C1R1"):
        predictions[cell] = {}
        for seed in SEEDS:
            run = summary["runs"][f"{cell}-seed{seed}"]
            folder = ROOT / run["folder"]
            for filename, expected in run["source_sha256"].items():
                require(sha(folder / filename) == expected, f"Audited evidence changed: {folder / filename}")
            require(read(folder / "prepared_sha256.json") == manifest["prepared_sha256"], "Run input mismatch")
            p = load_predictions(folder / "val_predictions.csv", ids, codes)
            require((p[prior] == 0).all(), "Known-prior probability not zero")
            predictions[cell][seed] = p
            evidence_sha[str((folder / "val_predictions.csv").relative_to(ROOT))] = sha(folder / "val_predictions.csv")

    positive_rows, overtakes, condition_rows, reports = [], [], [], {}
    max_reconstruction_error = 0.0
    for contrast, (before_cell, after_cell) in COMPARISONS.items():
        comparison_positives, per_seed = [], []
        for seed in SEEDS:
            before, after = predictions[before_cell][seed], predictions[after_cell][seed]
            macro_delta, seed_contribution, total_ap_before, total_ap_after = 0.0, 0.0, 0.0, 0.0
            for j, code in enumerate(codes):
                truth, eligible = labels[:, j], ~prior[:, j]
                positions = np.flatnonzero(truth)
                positive_count = len(positions)
                ap_a = float(average_precision_score(truth, before[:, j])) if positive_count else 0.0
                ap_b = float(average_precision_score(truth, after[:, j])) if positive_count else 0.0
                states_before, states_after = [], []
                condition_positive_rows = []
                for i in positions:
                    sa = threshold_state(before[:, j], truth, eligible, i)
                    sb = threshold_state(after[:, j], truth, eligible, i)
                    states_before.append(sa["threshold_precision"])
                    states_after.append(sb["threshold_precision"])
                    ap_contribution = (sb["threshold_precision"] - sa["threshold_precision"]) / positive_count
                    row = dict(contrast=contrast, seed=seed, patient_id=ids[i], **context[ids[i]], code=code,
                               description=descriptions[code], n_positive=positive_count)
                    for key in sa:
                        row[f"{key}_before"], row[f"{key}_after"] = sa[key], sb[key]
                        row[f"{key}_delta"] = sb[key] - sa[key]
                    row.update(AP_contribution_delta=float(ap_contribution),
                               mAP_contribution_delta=float(ap_contribution / n_targets),
                               positive_Brier_delta=float((1 - sb["probability"])**2 - (1 - sa["probability"])**2),
                               positive_BCE_delta=float(-np.log(np.clip(sb["probability"], 1e-7, 1 - 1e-7)) +
                                                       np.log(np.clip(sa["probability"], 1e-7, 1 - 1e-7))))
                    old_above = before[:, j] > before[i, j]
                    new_above = after[:, j] > after[i, j]
                    new_tied = after[:, j] == after[i, j]
                    new_strict = np.flatnonzero(eligible & ~truth & new_above & ~old_above)
                    new_ties = np.flatnonzero(eligible & ~truth & new_tied & (before[:, j] < before[i, j]))
                    row.update(newly_strictly_above_negative_count=int(len(new_strict)),
                               newly_tied_from_below_negative_count=int(len(new_ties)))
                    for negative_index, kind in [(int(x), "newly_strictly_above") for x in new_strict] + [(int(x), "newly_tied_from_below") for x in new_ties]:
                        k = negative_index
                        old_relation = "tie" if before[k, j] == before[i, j] else "below"
                        overtakes.append(dict(contrast=contrast, seed=seed, code=code, description=descriptions[code],
                                              positive_patient_id=ids[i], positive_alias=aliases[ids[i]],
                                              negative_patient_id=ids[k], negative_alias=aliases[ids[k]],
                                              positive_history_group=context[ids[i]]["history_group"],
                                              negative_history_group=context[ids[k]]["history_group"],
                                              transition=kind, relation_before=old_relation,
                                              positive_probability_before=float(before[i, j]), positive_probability_after=float(after[i, j]),
                                              negative_probability_before=float(before[k, j]), negative_probability_after=float(after[k, j]),
                                              positive_probability_delta=float(after[i, j] - before[i, j]),
                                              negative_probability_delta=float(after[k, j] - before[k, j]),
                                              negative_minus_positive_margin_before=float(before[k, j] - before[i, j]),
                                              negative_minus_positive_margin_after=float(after[k, j] - after[i, j])))
                    condition_positive_rows.append(row)
                precision_a = float(np.mean(states_before)) if positive_count else 0.0
                precision_b = float(np.mean(states_after)) if positive_count else 0.0
                error = max(abs(precision_a - ap_a), abs(precision_b - ap_b),
                            abs(sum(r["AP_contribution_delta"] for r in condition_positive_rows) - (ap_b - ap_a)))
                require(error < 1e-12, f"Tie-aware AP reconstruction failed: {contrast}/{seed}/{code}")
                max_reconstruction_error = max(max_reconstruction_error, error)
                condition_rows.append(dict(contrast=contrast, seed=seed, code=code, description=descriptions[code],
                                           n_positive=positive_count, AP_before=ap_a, AP_after=ap_b,
                                           AP_delta=ap_b - ap_a, mAP_contribution_delta=(ap_b - ap_a) / n_targets,
                                           reconstructed_AP_before=precision_a, reconstructed_AP_after=precision_b,
                                           positive_probability_mean_before=float(before[truth, j].mean()),
                                           positive_probability_mean_after=float(after[truth, j].mean()),
                                           at_risk_negative_probability_mean_before=float(before[eligible & ~truth, j].mean()),
                                           at_risk_negative_probability_mean_after=float(after[eligible & ~truth, j].mean()),
                                           positive_probability_rose_but_rank_worsened=sum(r["probability_delta"] > 0 and r["rank_worst_delta"] > 0 for r in condition_positive_rows),
                                           positive_rank_worsened=sum(r["rank_worst_delta"] > 0 for r in condition_positive_rows),
                                           positive_precision_decreased=sum(r["threshold_precision_delta"] < 0 for r in condition_positive_rows),
                                           new_strict_overtake_pairs=sum(r["newly_strictly_above_negative_count"] for r in condition_positive_rows),
                                           new_tie_pairs=sum(r["newly_tied_from_below_negative_count"] for r in condition_positive_rows)))
                macro_delta += (ap_b - ap_a) / n_targets
                total_ap_before += ap_a / n_targets
                total_ap_after += ap_b / n_targets
                seed_contribution += sum(r["mAP_contribution_delta"] for r in condition_positive_rows)
                comparison_positives.extend(condition_positive_rows)
            require(abs(macro_delta - seed_contribution) < 1e-12, "mAP contribution identity failed")
            require(abs(total_ap_before - summary["runs"][f"{before_cell}-seed{seed}"]["summary"]["best_validation_mAP"]) < 1e-12
                    and abs(total_ap_after - summary["runs"][f"{after_cell}-seed{seed}"]["summary"]["best_validation_mAP"]) < 1e-12,
                    "CSV reconstructed mAP differs from audited saved score")
            per_seed.append(dict(seed=seed, mAP_delta=float(macro_delta), reconstructed_mAP_delta=float(seed_contribution)))
        positive_rows.extend(comparison_positives)
        cells, patients = aggregate_positive_rows(comparison_positives, n_targets)
        condition_summary = []
        for code in codes:
            rows = [r for r in condition_rows if r["contrast"] == contrast and r["code"] == code]
            condition_summary.append(dict(code=code, description=descriptions[code], n_positive=rows[0]["n_positive"],
                                          mean_AP_before=float(np.mean([r["AP_before"] for r in rows])),
                                          mean_AP_after=float(np.mean([r["AP_after"] for r in rows])),
                                          mean_AP_delta=float(np.mean([r["AP_delta"] for r in rows])),
                                          mean_mAP_contribution=float(np.mean([r["mAP_contribution_delta"] for r in rows])),
                                          negative_AP_delta_seeds=sum(r["AP_delta"] < 0 for r in rows), per_seed=rows))
        condition_summary.sort(key=lambda r: r["mean_mAP_contribution"])
        worse_rows = [r for r in comparison_positives if r["threshold_precision_delta"] < 0]
        selected_keys = {(r["patient_id"], r["code"]) for r in cells[:30]}
        exported_pair_scope = {"type": "all_new_overtake_pairs", "filter": None}
        if contrast in READOUT_COMPARISONS:
            selected_codes = [r["code"] for r in condition_summary[:5]]
            selected_keys = {(r["patient_id"], code) for code in selected_codes
                             for r in [x for x in cells if x["code"] == code][:3]}
            exported_pair_scope = {
                "type": "selected_readout_case_pairs",
                "filter": "Five targets with most negative mean AP contribution; three positive patient cells with most negative mean contribution within each target; all newly overtaking pairs for these cases in each seed.",
                "condition_codes": selected_codes,
                "positive_patient_condition_cells": [{"patient_id": p, "patient_alias": aliases[p], "code": c}
                                                     for p, c in sorted(selected_keys)],
            }
            overtakes = [r for r in overtakes if r["contrast"] != contrast
                         or (r["positive_patient_id"], r["code"]) in selected_keys]
        exported_pair_scope["exported_pair_count"] = sum(r["contrast"] == contrast for r in overtakes)
        relevant_overtakes = [r for r in overtakes if r["contrast"] == contrast
                             and (r["positive_patient_id"], r["code"]) in selected_keys]
        case_examples, recurrent = [], []
        for patient_id, code in sorted(selected_keys):
            case_rows = [r for r in relevant_overtakes if r["positive_patient_id"] == patient_id and r["code"] == code]
            for seed in SEEDS:
                seed_rows = [r for r in case_rows if r["seed"] == seed]
                seed_rows.sort(key=lambda r: r["negative_probability_after"], reverse=True)
                case_examples.extend(seed_rows[:5])
            by_negative = {}
            for r in case_rows:
                by_negative.setdefault(r["negative_patient_id"], []).append(r)
            for negative_id, values in by_negative.items():
                if len(values) >= 2:
                    recurrent.append(dict(positive_patient_id=patient_id, positive_alias=aliases[patient_id], code=code,
                                          negative_patient_id=negative_id, negative_alias=aliases[negative_id],
                                          n_seed_new_overtakes=len(values), per_seed=values))
        recurrent.sort(key=lambda r: (-r["n_seed_new_overtakes"], r["positive_alias"], r["code"]))
        negative_contribution_rows = [r for r in comparison_positives if r["mAP_contribution_delta"] < 0]
        negative_components = {
            "all_negative_threshold_contributions": float(sum(r["mAP_contribution_delta"] for r in negative_contribution_rows) / len(SEEDS)),
            "negative_threshold_contributions_with_probability_rise": float(sum(r["mAP_contribution_delta"] for r in negative_contribution_rows if r["probability_delta"] > 0) / len(SEEDS)),
            "negative_threshold_contributions_with_more_negative_patients_above": float(sum(r["mAP_contribution_delta"] for r in negative_contribution_rows if r["at_risk_negative_strictly_above_delta"] > 0) / len(SEEDS)),
            "positive_threshold_contributions": float(sum(r["mAP_contribution_delta"] for r in comparison_positives if r["mAP_contribution_delta"] > 0) / len(SEEDS)),
            "interpretation": "Signed additive mAP contribution sums, averaged over fitted seeds. These are not causal patient interventions; positive-positive reordering can move individual threshold contributions while total AP is unchanged.",
        }
        reports[contrast] = dict(before_cell=before_cell, after_cell=after_cell, per_seed=per_seed,
                                 mean_single_model_mAP_delta=float(np.mean([r["mAP_delta"] for r in per_seed])),
                                 condition_ranked_contributions=condition_summary,
                                 worst_positive_cells=cells[:40], best_positive_cells=list(reversed(cells[-10:])),
                                 worst_patients_aggregate=patients[:30],
                                 all_patient_aggregate=patients,
                                 key_case_overtakes=case_examples,
                                 recurrent_key_case_overtakers=recurrent,
                                 negative_overtake_CSV_scope=exported_pair_scope,
                                 threshold_contribution_partitions=negative_components,
                                 descriptive_counts=dict(positive_patient_condition_seed_rows=len(comparison_positives),
                                    threshold_precision_decreased=len(worse_rows),
                                    probability_rose_among_precision_decreased=sum(r["probability_delta"] > 0 for r in worse_rows),
                                    rank_worsened=len([r for r in comparison_positives if r["rank_worst_delta"] > 0]),
                                    probability_rose_among_rank_worsened=sum(r["probability_delta"] > 0 and r["rank_worst_delta"] > 0 for r in comparison_positives),
                                    newly_strict_overtake_pairs=sum(r["newly_strictly_above_negative_count"] for r in comparison_positives),
                                    newly_tied_from_below_pairs=sum(r["newly_tied_from_below_negative_count"] for r in comparison_positives)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=1, purpose="Exact saved-validation patient attribution of AP changes, not mechanistic or clinical causation.",
                  seeds=list(SEEDS), uses_probability_ensemble=False, cohort=dict(n_patients=n, n_targets=n_targets, n_positive_cells=int(labels.sum())),
                  comparisons=reports, alignment="Exact original CSV patient and CODE order required; no silent reordering.",
                  patient_alias_rule="V001..V365 follow frozen validation array row order; IDs are local synthetic-data patient UUIDs.",
                  patient_metadata={patient: dict(patient_id=patient, **context[patient]) for patient in ids},
                  definitions={"threshold_precision": "# positives with score>=this positive score / # all patients with score>=this positive score",
                      "AP": "Mean threshold precision over all positive patients, including tied positives; exactly sklearn average_precision_score.",
                      "patient_AP_contribution": "Threshold precision / positive count for that target; delta sums exactly to target AP delta.",
                      "patient_mAP_contribution": "Patient AP contribution / all 40 target count; sum equals mAP delta.",
                      "rank_best": "1 + count(scores > patient score)", "rank_worst": "count(scores >= patient score)",
                      "newly_strictly_above": "At-risk negative was below or tied before, strictly above after.",
                      "newly_tied_from_below": "At-risk negative was strictly below before and tied after.",
                      "patient_age": "Derived from frozen standardized age and train-only age_mean/age_scale; approximate float32 inversion; no names or birthdates."},
                  checks=dict(frozen_sources_and_data=True, strict_audit_controls=True, prior_probability_zero=True,
                              exact_AP_and_mAP_reconstruction=True, max_AP_reconstruction_error=float(max_reconstruction_error)),
                  input_sha256=dict(analysis_script=sha(Path(__file__)), frozen_manifest=sha(AUDIT / "frozen_manifest.json"), audit_summary=sha(AUDIT / "audit_summary.json"),
                                    predictions=evidence_sha),
                  interpretation_limits=["AP attribution is algebraic: it locates changed ranking contributions, not the medical reason the model assigned scores.",
                     "Negative overtake CSV is exhaustive for attention contrasts; readout contrasts are restricted to declared top-five-target/top-three-victim cases. Positive and condition AP decomposition remains exhaustive for all four contrasts.",
                     "A newly overtaking negative is a descriptive pair; AP changes are nonlinear, so no arbitrary per-pair AP causal contribution is assigned.",
                     "Ranks, thresholds and metrics are calculated per fitted seed first; mean probability fields describe score scales and are never used to rank or evaluate an ensemble.",
                     "A positive score may rise while rank/precision worsens if competing patients rise more; a lower rank can also be partly positive-positive reordering.",
                     "Full rankings include known-prior zero-scored negatives; at-risk ranks are reported separately.",
                     "Overlapping conditions share patients and are not independent clinical replications; patient-aggregate contributions intentionally follow the official macro metric.",
                     "Fixed validation and selected checkpoints were reused; findings are descriptive and do not estimate unbiased test generalization."])
    write_json(OUTPUT / "patient_ranking.json", result)
    write_csv(OUTPUT / "positive_rank_changes.csv", positive_rows)
    write_csv(OUTPUT / "negative_overtakes.csv", overtakes)
    write_csv(OUTPUT / "condition_rank_changes.csv", condition_rows)
    print(json.dumps(dict(passed=True, max_AP_reconstruction_error=max_reconstruction_error,
                          positive_rows=len(positive_rows), negative_overtake_rows=len(overtakes),
                          output_dir=str(OUTPUT)), ensure_ascii=False))


if __name__ == "__main__":
    main()
