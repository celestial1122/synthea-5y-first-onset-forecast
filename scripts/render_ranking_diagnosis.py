"""Join read-only ranking diagnostics and render figures; never fit a model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_probability_rank_bridge as bridge

OUT = ROOT / "artifacts/ranking_diagnosis"
AUDIT = ROOT / "artifacts/experimental_audit_strict"
PREPARED = ROOT / "artifacts/optimization/prepared_l512"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, filename, caption, top=.95):
    fig.text(.01, .01, caption, fontsize=8, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, .07, 1, top))
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def main():
    scores = pd.read_csv(OUT / "condition_score_components.csv", dtype={"code": str})
    positives = pd.read_csv(OUT / "positive_rank_changes.csv", dtype={"code": str})
    patient_data = read(OUT / "patient_ranking.json")
    swap = read(OUT / "probability_rank_bridge.json")
    metadata = read(PREPARED / "metadata.json")
    codes = metadata["target_codes"]
    with np.load(PREPARED / "val.npz", allow_pickle=False) as z:
        ids = z["patient_ids"].astype(str).tolist()
        y, prior = z["labels"].astype(float), z["prior_conditions"].astype(bool)
    all_predictions = {(cell, seed): bridge.load_predictions(cell, seed, ids, codes)[0]
                       for cell in ("C0R0", "C1R0", "C0R1", "C1R1") for seed in bridge.SEEDS}
    # Patient contributions use the same global denominators as saved metrics.
    loss_rows = []
    positive_mask, negative_mask = y.astype(bool), (~prior & ~y.astype(bool))
    for name, (a, b) in bridge.CONTRASTS.items():
        per_seed = []
        for seed in bridge.SEEDS:
            old, new = [bridge.losses(y, all_predictions[cell, seed]) for cell in (a, b)]
            delta = {key: new[key]-old[key] for key in ("BCE", "Brier")}
            per_seed.append(delta)
        for i, patient in enumerate(ids):
            pm = patient_data["patient_metadata"][patient]
            row = {"contrast": name, "patient_id": patient, "patient_alias": pm["patient_alias"],
                   "history_group": pm["history_group"], "full_event_count": pm["full_event_count"],
                   "dropped_event_count": pm["dropped_event_count"]}
            for metric in ("BCE", "Brier"):
                denominator = int((~prior).sum()) if metric == "BCE" else y.size
                for part, mask in (("positive", positive_mask), ("eligible_negative", negative_mask)):
                    value = np.mean([d[metric][i, mask[i]].sum()/denominator for d in per_seed])
                    row[f"{metric}_{part}_global_contribution"] = float(value)
                row[f"{metric}_total_global_contribution"] = sum(row[f"{metric}_{p}_global_contribution"]
                                                                 for p in ("positive", "eligible_negative"))
            loss_rows.append(row)
    with (OUT / "patient_loss_changes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loss_rows[0])); writer.writeheader(); writer.writerows(loss_rows)
    for name in bridge.CONTRASTS:
        for metric in ("BCE", "Brier"):
            actual = sum(r[f"{metric}_total_global_contribution"] for r in loss_rows if r["contrast"] == name)
            expected = swap["contrasts"][name]["mean_single_model"][metric]["all_eligible"]["total_delta"]
            if not np.isclose(actual, expected, atol=1e-12, rtol=0):
                raise AssertionError("Patient proper-loss contributions failed reconstruction")

    selected = [
        ("V085", "723857007", "C1R0_minus_C0R0", "Brain micro-haemorrhage"),
        ("V211", "254637007", "C1R0_minus_C0R0", "Non-small cell lung cancer"),
        ("V077", "162864005", "C0R1_minus_C0R0", "Obesity"),
        ("V338", "424132000", "C1R0_minus_C0R0", "Lung cancer stage 1"),
    ]
    cases = []
    for alias, code, contrast, label in selected:
        frame = positives[(positives.patient_alias == alias) & (positives.code == code)
                          & (positives.contrast == contrast)].sort_values("seed")
        if len(frame) != 3:
            raise AssertionError("A case must include all three fitted seeds")
        records = frame.to_dict(orient="records")
        patient = records[0]["patient_id"]
        relevant_pairs = []
        with (OUT / "negative_overtakes.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["contrast"] == contrast and row["positive_patient_id"] == patient and row["code"] == code:
                    relevant_pairs.append(row)
        relevant_pairs.sort(key=lambda r: -float(r["negative_probability_after"]))
        cases.append({"alias": alias, "patient_id": patient, "code": code, "label": label,
                      "contrast": contrast, "metadata": patient_data["patient_metadata"][patient],
                      "per_seed": records, "new_overtakers_top_by_new_probability": relevant_pairs[:15]})
    write(OUT / "key_cases.json", {"scope": "Post-hoc descriptive cases; no model/clinical intervention.",
                                     "cases": cases})

    # Per-condition changes: losses and AP need not move together within a class.
    chosen = ["424132000", "162573006", "254637007", "723857007", "22325002", "444814009", "195662009", "10509002"]
    labels = ["Lung cancer stage 1 (9+)", "Suspected lung cancer (10+)", "Non-small cell lung cancer (9+)",
              "Brain micro-haemorrhage (11+)", "Abnormal gait (11+)", "Viral sinusitis (99+)",
              "Viral pharyngitis (72+)", "Acute bronchitis (61+)"]
    frame = scores[(scores.contrast == "attention_at_legacy") & (scores.group == "overall")]
    means = frame.groupby("code", sort=False).mean(numeric_only=True).loc[chosen]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    pos = np.arange(len(chosen))
    for ax, values, xlabel in zip(axes, [means.AP_delta, means.brier_delta, means.bce_new-means.bce_old],
                                ["Change in class AP (higher is better)", "Change in class Brier (lower is better)",
                                 "Change in class at-risk BCE (lower is better)"]):
        ax.barh(pos, values, height=.55, color="steelblue")
        ax.axvline(0, color="black", lw=.8); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_yticks(pos, labels); axes[0].invert_yaxis()
    fig.suptitle("Attention swap C1R0 - C0R0: class ranking and probability error can disagree")
    save(fig, "disease_tradeoffs.png", "Source: official validation n=365; mean of three single-model changes at mAP-selected checkpoints. '+' in labels = positive patient count. Post-hoc classes.")

    # Exact descriptive cross-score experiment. No probability vector is deployed.
    item = swap["contrasts"]["attention_R0"]
    a_ap = float(np.mean([r["mAP_old"] for r in item["per_seed"]]))
    b_ap = float(np.mean([r["mAP_new"] for r in item["per_seed"]]))
    data = item["mean_single_model"]
    states = [("Original output", a_ap, "old"), ("New output", b_ap, "new"),
              ("Old order + new score values", a_ap, "old_order_new_values"),
              ("New order + old score values", b_ap, "new_order_old_values")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric in zip(axes, ("BCE", "Brier")):
        for (label, ap, key), marker, color in zip(states, ("o", "s", "^", "x"),
                                                  ("black", "steelblue", "darkorange", "gray")):
            ax.scatter(ap, data[metric]["all_eligible"][key], label=label, marker=marker, c=color, s=65)
        ax.set(xlabel="Mean single-model mAP (higher is better)",
               ylabel=f"{'At-risk ' if metric == 'BCE' else ''}{metric} (lower is better)",
               title=f"Ordering / score-distribution swap: {metric}")
        ax.grid(alpha=.15); ax.ticklabel_format(useOffset=False)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2, fontsize=9)
    save(fig, "probability_rank_swap.png", "Source: fixed validation n=365; three-seed metric means. Diagnostic crossed predictions only; no fitting, recalibration, model selection or causal network attribution.", top=.83)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9), sharey=True)
    for ax, case in zip(axes, cases):
        for row, color in zip(case["per_seed"], ("steelblue", "darkorange", "seagreen")):
            ax.plot([0, 1], [row["rank_best_before"], row["rank_best_after"]], "o-", color=color,
                    label=f"Seed {row['seed']}")
        ax.set_xticks([0, 1], ["Before", "After"])
        ax.set_title(f"{case['alias']}: {case['label']}\n{case['contrast'].replace('_minus_', ' - ')}", fontsize=10)
        ax.set_yscale("log"); ax.set_ylim(400, .8); ax.grid(axis="y", alpha=.15)
        ax.set_yticks([1, 5, 10, 30, 100, 365], ["1", "5", "10", "30", "100", "365"])
    axes[0].set_ylabel("Patient rank within condition (1 = highest; log axis)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Actual positive patients: dropped leading cases and rescued low-scoring cases")
    save(fig, "patient_rank_cases.png", "Source: saved validation probabilities; no ties at positive thresholds. V077 changes the readout; other panels change attention. Post-hoc cases, not independent evidence.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, seed in zip(axes, bridge.SEEDS):
        for cell, color in (("C0R0", "black"), ("C1R0", "steelblue")):
            name = f"{cell}-seed{seed}" + ("-repeat-a" if cell == "C0R0" else "")
            folder = AUDIT / "runs" / name
            rows = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
            ax.plot([r["epoch"] for r in rows], [r["val/loss"] for r in rows], label=cell, color=color)
            selected_epoch = read(folder / "summary.json")["best_epoch"]
            selected_row = next(r for r in rows if r["epoch"] == selected_epoch)
            minimum = min(rows, key=lambda r: r["val/loss"])
            ax.scatter(selected_epoch, selected_row["val/loss"], color=color, marker="o", s=40)
            ax.scatter(minimum["epoch"], minimum["val/loss"], color=color, marker="x", s=65)
        ax.set(xlabel="Training epoch", title=f"Seed {seed}")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Validation at-risk BCE (lower is better)"); axes[-1].legend()
    fig.suptitle("Different mAP-selected epochs change the reported BCE: circle = selected; cross = BCE minimum")
    save(fig, "selection_tradeoff.png", "Source: frozen history.jsonl, 12-19 recorded epochs. This compares recorded trajectories; it does not replace checkpoint selection or identify causal training-stage effects.")

    summary = {"schema_version": 1, "scope": "Saved strict 2x2 outputs; no new model or checkpoint selection.",
               "seeds": list(bridge.SEEDS), "patient_proper_loss_reconstruction_passed": True,
               "primary_contrast": "C1R0_minus_C0R0", "key_cases": cases,
               "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                   *[ROOT / "scripts" / n for n in ["analyze_probability_rank_bridge.py", "analyze_patient_ranking.py",
                                                    "analyze_score_components.py", "analyze_selection_tradeoff.py"]]]},
               "diagnostic_input_sha256": {n: sha(OUT/n) for n in ["score_components.json", "patient_ranking.json",
                        "selection_tradeoff.json", "probability_rank_bridge.json", "positive_rank_changes.csv"]}}
    write(OUT / "diagnosis_summary.json", summary)
    print(json.dumps({"checks": "passed", "patient_loss_rows": len(loss_rows), "key_cases": len(cases), "figures": 4}))


if __name__ == "__main__":
    main()
