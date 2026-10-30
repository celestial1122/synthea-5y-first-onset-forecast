"""Descriptive score-distribution/order swaps; no fit, calibration or selection.

For each condition, the cross-world vectors transplant the sorted eligible
probabilities of one model onto the patient order of the other. With distinct
scores, AP follows only order. A symmetric two-path decomposition of proper
losses is algebraically exact, but is not a causal attribution to a network
mechanism. Prior diagnoses stay at zero throughout.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREPARED = ROOT / "artifacts/optimization/prepared_l512"
AUDIT = ROOT / "artifacts/experimental_audit_strict"
OUTPUT = ROOT / "artifacts/ranking_diagnosis"
SEEDS = (17, 42, 73)
CONTRASTS = {"attention_R0": ("C0R0", "C1R0"), "attention_R1": ("C0R1", "C1R1"),
             "readout_C0": ("C0R0", "C0R1"), "readout_C1": ("C1R0", "C1R1")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def transplant(old, new):
    """Return new values/old order and old values/new order, refusing ties."""
    old, new = np.asarray(old, float), np.asarray(new, float)
    if old.shape != new.shape or old.ndim != 1 or len(old) < 1:
        raise ValueError("Aligned nonempty vectors are required")
    if (not np.isfinite(old).all() or not np.isfinite(new).all()
            or np.any(old <= 0) or np.any(new <= 0) or np.any(old >= 1) or np.any(new >= 1)):
        raise ValueError("Finite probabilities strictly inside (0,1) are required")
    if len(np.unique(old)) != len(old) or len(np.unique(new)) != len(new):
        raise ValueError("Ties require another estimand; do not break them arbitrarily")
    old_order = np.argsort(old, kind="stable")
    new_order = np.argsort(new, kind="stable")
    old_order_new_values = np.empty_like(old)
    new_order_old_values = np.empty_like(new)
    old_order_new_values[old_order] = new[new_order]
    new_order_old_values[new_order] = old[old_order]
    return old_order_new_values, new_order_old_values


def symmetric_effects(base, variant, old_order_new_values, new_order_old_values):
    distribution = .5 * ((old_order_new_values - base) + (variant - new_order_old_values))
    order = .5 * ((new_order_old_values - base) + (variant - old_order_new_values))
    if not np.isclose(distribution + order, variant - base, atol=1e-12, rtol=0):
        raise AssertionError("Decomposition does not reconstruct the loss difference")
    return {"total_delta": float(variant - base), "score_distribution_effect": float(distribution),
            "patient_order_effect": float(order)}


def losses(y, p):
    raw = np.asarray(p, float)
    clipped = np.clip(raw, 1e-7, 1 - 1e-7)
    return {"BCE": -(y * np.log(clipped) + (1-y) * np.log1p(-clipped)), "Brier": (raw-y)**2}


def load_predictions(cell, seed, ids, codes):
    name = f"{cell}-seed{seed}" + ("-repeat-a" if cell == "C0R0" else "")
    path = AUDIT / "runs" / name / "val_predictions.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0] != ["patient_id", *codes] or [row[0] for row in rows[1:]] != ids:
        raise ValueError(f"Prediction alignment changed: {path}")
    p = np.asarray([row[1:] for row in rows[1:]], float)
    if p.shape != (len(ids), len(codes)) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid prediction probabilities")
    return p, path


def analyze():
    manifest = read(AUDIT / "frozen_manifest.json")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in manifest["prepared_sha256"].items():
        if sha(PREPARED / name) != expected:
            raise ValueError(f"Frozen data changed: {name}")
    if not read(AUDIT / "audit_summary.json")["factorial_controls"]["passed"]:
        raise ValueError("Factorial controls must pass")
    with np.load(PREPARED / "val.npz", allow_pickle=False) as z:
        ids, y, prior = z["patient_ids"].astype(str).tolist(), z["labels"].astype(float), z["prior_conditions"].astype(bool)
    metadata = read(PREPARED / "metadata.json")
    codes, descriptions = metadata["target_codes"], metadata["target_descriptions"]
    eligible = ~prior
    if np.any(y[prior]):
        raise AssertionError("Incident positives cannot be prior diagnoses")
    score_cache, input_hashes = {}, {}
    for cell in ("C0R0", "C1R0", "C0R1", "C1R1"):
        for seed in SEEDS:
            p, path = load_predictions(cell, seed, ids, codes)
            if np.any(p[prior] != 0):
                raise AssertionError("Known prior probabilities must be zero")
            score_cache[cell, seed] = p
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
    rows, contrasts = [], {}
    for name, (old_cell, new_cell) in CONTRASTS.items():
        runs = []
        for seed in SEEDS:
            a, b = score_cache[old_cell, seed], score_cache[new_cell, seed]
            ab, ba = np.zeros_like(a), np.zeros_like(a)
            ap_old, ap_new = [], []
            for j, code in enumerate(codes):
                mask = eligible[:, j]
                ab[mask, j], ba[mask, j] = transplant(a[mask, j], b[mask, j])
                ap = [float(average_precision_score(y[:, j], p[:, j])) for p in (a, b, ab, ba)]
                if not np.isclose(ap[0], ap[2], atol=1e-12, rtol=0) or not np.isclose(ap[1], ap[3], atol=1e-12, rtol=0):
                    raise AssertionError("Score transplant changed the supplied patient ordering AP")
                ap_old.append(ap[0]); ap_new.append(ap[1])
                item = {"contrast": name, "seed": seed, "code": code, "description": descriptions[j],
                        "n_positive": int(y[:, j].sum()), "n_eligible": int(mask.sum()),
                        "AP_old": ap[0], "AP_new": ap[1], "AP_delta": ap[1]-ap[0]}
                component = [losses(y[:, j], p[:, j]) for p in (a, b, ab, ba)]
                for metric in ("BCE", "Brier"):
                    denominator = int(eligible.sum()) if metric == "BCE" else y.size
                    values = [float(c[metric][mask].sum() / denominator) for c in component]
                    effect = symmetric_effects(*values)
                    item.update({f"{metric}_{key}_global_contribution": value for key, value in effect.items()})
                rows.append(item)
            run = {"seed": seed, "mAP_old": float(np.mean(ap_old)), "mAP_new": float(np.mean(ap_new)),
                   "mAP_delta": float(np.mean(ap_new)-np.mean(ap_old)), "AP_order_invariance_verified": True,
                   "metrics": {}}
            component = [losses(y, p) for p in (a, b, ab, ba)]
            for metric in ("BCE", "Brier"):
                denominator = int(eligible.sum()) if metric == "BCE" else y.size
                run["metrics"][metric] = {}
                for group, mask in (("all_eligible", eligible), ("positive", y.astype(bool)),
                                    ("eligible_negative", eligible & ~y.astype(bool))):
                    values = [float(c[metric][mask].sum() / denominator) for c in component]
                    effect = symmetric_effects(*values)
                    run["metrics"][metric][group] = {"old": values[0], "new": values[1],
                        "old_order_new_values": values[2], "new_order_old_values": values[3], **effect}
                overall = run["metrics"][metric]["all_eligible"]
                for key in ("total_delta", "score_distribution_effect", "patient_order_effect"):
                    reconstructed = sum(run["metrics"][metric][g][key] for g in ("positive", "eligible_negative"))
                    if not np.isclose(reconstructed, overall[key], atol=1e-12, rtol=0):
                        raise AssertionError("Positive/negative decomposition failed")
            runs.append(run)
        aggregate = {}
        for metric in ("BCE", "Brier"):
            aggregate[metric] = {}
            for group in ("all_eligible", "positive", "eligible_negative"):
                aggregate[metric][group] = {key: float(np.mean([r["metrics"][metric][group][key] for r in runs]))
                                          for key in runs[0]["metrics"][metric][group]}
        contrasts[name] = {"old": old_cell, "new": new_cell, "per_seed": runs, "mean_single_model": aggregate}
    report = {"schema_version": 1, "seeds": list(SEEDS), "n_patients": len(ids), "n_targets": len(codes),
              "uses_probability_ensemble": False, "fitted_or_selected_model": False,
              "decomposition": "Symmetric two-path (Shapley) descriptive swap of per-condition eligible score distribution and patient ordering.",
              "limitations": "Crossed predictions are diagnostic counterfactuals, not deployable models. The algebraic components are not unique causal effects. Probability distributions may reflect calibration, confidence, different checkpoints and patient mixtures; this analysis does not identify a neural mechanism. All estimates condition on the fixed validation patients and fitted checkpoints.",
              "BCE_denominator": int(eligible.sum()), "Brier_denominator": y.size,
              "known_prior_loss_delta": 0., "exact_eligible_ties": 0,
              "contrasts": contrasts, "input_sha256": input_hashes,
              "source_sha256": sha(Path(__file__))}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / "probability_rank_bridge.json", report)
    with (OUTPUT / "probability_rank_bridge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({name: item["mean_single_model"] for name, item in contrasts.items()}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    analyze()
