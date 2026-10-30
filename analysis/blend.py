"""Blend analysis on official val: transformer (strict C0R0, 3 seeds), GBM+LR tabular, dense transformer."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from ehr_load import load
d = load(); pats, Y, P, codes = d["pats"], d["Y"], d["P"], d["codes"]
val_ids = pats[pats.split == "val"].PATIENT.values
yv = Y.loc[val_ids].values; pv = P.loc[val_ids].values.astype(bool)
HERE = Path(__file__).parent

def score(p, name):
    p = np.where(pv, 0, p)
    aps = [average_precision_score(yv[:, j], p[:, j]) if yv[:, j].sum() else 0 for j in range(40)]
    aucs = [roc_auc_score(yv[:, j], p[:, j]) for j in range(40)]
    print(f"{name:60s} mAP={np.mean(aps):.4f}  macroAUROC={np.mean(aucs):.4f}")
    return np.array(aps)

# transformer C0R0 strict (project's own val predictions)
tf = []
for f in sorted(glob.glob(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\artifacts\experimental_audit_strict\runs\C0R0-seed*-repeat-a\val_predictions.csv")):
    df = pd.read_csv(f, dtype={"patient_id": str}).set_index("patient_id").reindex(val_ids)
    tf.append(df[codes].values)
tf_mean = np.mean(tf, 0)
for i, t in enumerate(tf): score(t, f"transformer C0R0 seed{i}")
ap_tf = score(tf_mean, "transformer C0R0 3-seed prob ensemble")
# the submitted 6-member ensemble
sub = pd.read_csv(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\artifacts\optimization\validation_predictions_optimized.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(val_ids)
ap_sub = score(sub[codes].values, "Codex submitted 6-member ensemble (val)")
gbm = np.load(HERE / "tabular_val_probs_noanchor.npy")
ap_g = score(gbm, "tabular avg(GBM, LR) no anchor_year")
score((gbm + tf_mean) / 2, "blend: tabular + transformer-3seed (50/50)")
score((gbm + sub[codes].values) / 2, "blend: tabular + Codex 6-ensemble (50/50)")
res = {}
for mode in ("dense_runs/sparse", "dense_runs/dense", "dense_runs/dense_mix", "aux_runs/sparse", "aux_runs/multi", "aux_runs/wide", "aux_runs/multi_wide"):
    fs = sorted(glob.glob(str(HERE / f"{mode}_seed*_val.npy")))
    if not fs: continue
    ps = [np.load(f) for f in fs]
    for f, p in zip(fs, ps): score(p, f"{mode} {Path(f).stem}")
    pm = np.mean(ps, 0); res[mode] = pm
    score(pm, f"{mode} {len(ps)}-seed prob ensemble")
    score((pm + gbm) / 2, f"blend: {mode} ensemble + tabular (50/50)")
if "aux_runs/wide" in res:
    w = res["aux_runs/wide"]
    score((w + gbm + tf_mean) / 3, "blend: wide3 + tabular + C0R0-3seed (1/3 each)")
    # best GBM grid ensemble if available
    g2 = sorted(glob.glob(str(HERE / "gbm_*_val.npy")))
    if g2:
        ge = np.mean([np.load(f) for f in g2], 0); score(ge, f"GBM grid ensemble ({len(g2)} configs)")
        score((ge + tf_mean) / 2, "blend: GBM grid ensemble + C0R0-3seed"); score((ge + w) / 2, "blend: GBM grid ensemble + wide3"); score((ge + tf_mean + w) / 3, "blend: GBM grid + C0R0-3 + wide3")
    ap_w = score(w, "wide ensemble (again)")
    cmp = pd.DataFrame({"val_pos": yv.sum(0), "transformer3": ap_tf, "tabular": ap_g, "wide3": ap_w, "blend_tab_tf": score((gbm + tf_mean) / 2, "blend again")}, index=[d["targets"].set_index("CODE").DESCRIPTION[c][:34] for c in codes]).round(3)
    pd.set_option("display.width", 200); print(cmp.to_string())
