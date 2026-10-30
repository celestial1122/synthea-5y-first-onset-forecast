"""Honest evaluation of the assembled predictions: official validation split (365 patients) and train 5-fold cross-fitted
OOF (2,791 patients), macro AUROC and mAP under the task's convention (every patient scored; patients who already had
the condition before the anchor carry label 0 and prediction 0) and under the at-risk-only convention, by target group.
Writes report/metrics.json and report/per_condition_metrics.csv."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT / "analysis")); from ehr_load import load
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
ids = {s: pats[pats.split == s].PATIENT.values for s in ("train", "val")}
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
grp = np.array(["terminal" if c in TERMINAL else "accident" if c in ACCIDENT else "chain" for c in codes])
pred = {"val": pd.read_csv(ROOT / "analysis/final/val_predictions_final.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(ids["val"])[codes].values,
        "train_oof": np.load(ROOT / "analysis/final/oof_crossfit.npy")}
def metrics(S, s):
    Yv, Pv = Y.loc[ids[s]].values, P.loc[ids[s]].values.astype(bool); out = {}
    for conv in ("all_patients", "at_risk"):
        au, apv = [], []
        for j in range(40):
            if conv == "all_patients": y, p = Yv[:, j], np.where(Pv[:, j], 0, S[:, j])
            else: y, p = Yv[~Pv[:, j], j], S[~Pv[:, j], j]
            au.append(roc_auc_score(y, p)); apv.append(average_precision_score(y, p))
        au, apv = np.array(au), np.array(apv)
        out[conv] = {"macro_AUROC": float(au.mean()), "mAP": float(apv.mean()), **{f"mAP_{g}": float(apv[grp == g].mean()) for g in ("accident", "chain", "terminal")}, **{f"AUROC_{g}": float(au[grp == g].mean()) for g in ("accident", "chain", "terminal")}, "per_target_AP": apv.tolist(), "per_target_AUROC": au.tolist()}
    return out
res = {"val": metrics(pred["val"], "val"), "train_oof": metrics(pred["train_oof"], "train")}
# bootstrap CI of val mAP (patients resampled)
rng = np.random.default_rng(0); Yv, Pv, S = Y.loc[ids["val"]].values, P.loc[ids["val"]].values.astype(bool), pred["val"]; bs = []
for _ in range(300):
    i = rng.integers(0, len(S), len(S)); bs.append(np.mean([average_precision_score(Yv[i, j], np.where(Pv[i, j], 0, S[i, j])) if Yv[i, j].sum() else 0 for j in range(40)]))
res["val"]["mAP_bootstrap_95ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
(ROOT / "report").mkdir(exist_ok=True); json.dump(res, open(ROOT / "report/metrics.json", "w"), indent=1)
per = pd.DataFrame({"code": codes, "condition": [desc[c] for c in codes], "group": grp, "prevalence_train": [float(Y.loc[ids["train"]].values[:, j].mean()) for j in range(40)],
                    "AP_train_oof": res["train_oof"]["all_patients"]["per_target_AP"], "AUROC_train_oof": res["train_oof"]["all_patients"]["per_target_AUROC"],
                    "AP_val": res["val"]["all_patients"]["per_target_AP"], "AUROC_val": res["val"]["all_patients"]["per_target_AUROC"]}).set_index("code")
per.round(4).to_csv(ROOT / "report/per_condition_metrics.csv")
for s in ("train_oof", "val"):
    m = res[s]["all_patients"]; print(f"{s:9s} all-patients convention: macro AUROC {m['macro_AUROC']:.4f}  mAP {m['mAP']:.4f}  (accident {m['mAP_accident']:.3f} / chain {m['mAP_chain']:.3f} / terminal {m['mAP_terminal']:.3f})")
print("val mAP bootstrap 95% CI:", [round(x, 4) for x in res["val"]["mAP_bootstrap_95ci"]])
