"""Honest blend: choose the GBM/transformer weight on 2791-patient OOF predictions (both models out-of-fold),
report val once, write predictions_final.csv. Inputs: analysis/v{2,3}/tabular_{oof,val,test}.npy (pats order within split)
and analysis/cv_tf/<mode>_{oof,val,test}.csv (patient_id indexed)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from ehr_load import load
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"
TAB = sys.argv[1] if len(sys.argv) > 1 else "v3"; MODE = sys.argv[2] if len(sys.argv) > 2 else "multi_wide"
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]
desc = dict(zip(targets.CODE, targets.DESCRIPTION))
ids = {s: pats[pats.split == s].PATIENT.values for s in ("train", "val", "test")}
_pref = "" if TAB == "clean" else "tabular_"
tab = {s: np.load(HERE / TAB / f"{_pref}{k}.npy") for s, k in (("train", "oof"), ("val", "val"), ("test", "test"))}
tf = {s: pd.read_csv(HERE / "cv_tf" / f"{MODE}_{k}.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(ids[s])[codes].values for s, k in (("train", "oof"), ("val", "val"), ("test", "test"))}
Yv = {s: Y.loc[ids[s]].values for s in ("train", "val")}; Pv = {s: P.loc[ids[s]].values.astype(bool) for s in ("train", "val", "test")}
GROUPS = [["162573006", "424132000", "254637007"], ["22325002", "723857007"]]
def group_avg(p, Pm):
    p = p.copy()
    for grp in GROUPS:
        jj = [codes.index(c) for c in grp]; m = p[:, jj].mean(1, keepdims=True); p[:, jj] = np.where(Pm[:, jj], 0, m)
    return p
def mAP(p, s):
    p = np.where(Pv[s], 0, p); return float(np.mean([average_precision_score(Yv[s][:, j], p[:, j]) if Yv[s][:, j].sum() else 0 for j in range(40)]))
def auroc(p, s):
    p = np.where(Pv[s], 0, p); return float(np.mean([roc_auc_score(Yv[s][:, j], p[:, j]) for j in range(40)]))
print(f"tabular {TAB}:       OOF mAP={mAP(tab['train'],'train'):.4f}  val mAP={mAP(tab['val'],'val'):.4f}")
print(f"transformer {MODE}: OOF mAP={mAP(tf['train'],'train'):.4f}  val mAP={mAP(tf['val'],'val'):.4f}")
best = None
for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
    o = mAP(w * tab["train"] + (1 - w) * tf["train"], "train"); v = mAP(w * tab["val"] + (1 - w) * tf["val"], "val")
    print(f"  w_tabular={w:.1f}: OOF mAP={o:.4f}  val mAP={v:.4f}")
    if best is None or o > best[1]: best = (w, o)
w = best[0]; print(f"chosen on OOF: w_tabular={w:.1f} (OOF {best[1]:.4f})")
final = {s: group_avg(np.where(Pv[s], 0, w * tab[s] + (1 - w) * tf[s]), Pv[s]) for s in ("train", "val", "test")}
print(f"FINAL (OOF-chosen weight, dup-group avg): OOF mAP={mAP(final['train'],'train'):.4f}  val mAP={mAP(final['val'],'val'):.4f}  val macroAUROC={auroc(final['val'],'val'):.4f}")
pd.DataFrame(final["test"], index=pd.Index(ids["test"], name="patient_id"), columns=codes).to_csv(ROOT / "predictions_final.csv")
pd.DataFrame(final["val"], index=pd.Index(ids["val"], name="patient_id"), columns=codes).to_csv(HERE / "val_predictions_final.csv")
per = pd.DataFrame({"cond": [desc[c][:34] for c in codes], "oof_tab": [average_precision_score(Yv["train"][:, j], np.where(Pv["train"][:, j], 0, tab["train"][:, j])) for j in range(40)],
                    "oof_tf": [average_precision_score(Yv["train"][:, j], np.where(Pv["train"][:, j], 0, tf["train"][:, j])) for j in range(40)],
                    "oof_final": [average_precision_score(Yv["train"][:, j], final["train"][:, j]) for j in range(40)],
                    "val_final": [average_precision_score(Yv["val"][:, j], final["val"][:, j]) for j in range(40)]}, index=codes).round(3)
pd.set_option("display.width", 200); print(per.to_string()); per.to_csv(HERE / "per_condition_final.csv")
json.dump({"w_tabular": float(w), "oof_mAP": mAP(final["train"], "train"), "val_mAP": mAP(final["val"], "val"), "val_auroc": auroc(final["val"], "val")}, open(HERE / "summary_final.json", "w"), indent=1)
