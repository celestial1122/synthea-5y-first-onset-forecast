"""Reproduce the submitted predictions from the exported model artifacts (the Hugging Face bundle):
    python predict.py --export export            # writes predictions_reproduced.csv, compares with predictions.csv, evaluates on val
Requires the feature matrices built by `python reproduce.py features` (analysis/clean/*.parquet, analysis/ppds_gcr/).
Artifacts: export/A/<code>_fold<f>.json (+ _cols.npy), export/B/<code>_fold<f>.txt (+ _cols.npy), export/C/<code>.txt,
export/assembly/{weights.csv, assembly_meta.json, <member>_oof.npy}."""
import sys, json, argparse, numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT = Path(__file__).resolve().parent; AN = ROOT / "analysis"; sys.path.insert(0, str(AN)); from ehr_load import load
ap = argparse.ArgumentParser(); ap.add_argument("--export", default="export"); ap.add_argument("--splits", default="val,test"); args = ap.parse_args()
E = Path(args.export) if Path(args.export).is_absolute() else ROOT / args.export
meta = json.load(open(E / "assembly/assembly_meta.json")); names, codes = meta["candidates"], meta["codes"]; W = pd.read_csv(E / "assembly/weights.csv", index_col=0)
d = load(); pats, P = d["pats"], d["P"]; ids = {s: np.array(meta["ids"][s]) for s in ("train", "val", "test")}
splits = args.splits.split(","); Pv = {s: P.loc[ids[s]].values.astype(bool) for s in ("train", *splits)}
# ---- features
import os; os.environ["EHR_GENOMICS"] = "3"; import genfeat
Xc = genfeat.augment(pd.read_parquet(AN / "clean/X.parquet"))                      # member B features (clean + genomics + cost v2)
mA = json.load(open(E / "A/meta.json")); Xp = pd.read_parquet(AN / mA["ds"] / "X.parquet").set_index("PATIENT"); Mp = np.load(AN / mA["ds"] / "meta.npz", allow_pickle=True)
off = {s: (Mp["k"] == 0) & (Mp["split"] == s) for s in splits}
XA = {s: Xp.values[off[s]][pd.Series(np.arange(off[s].sum()), index=Mp["pid"][off[s]].astype(str)).reindex(ids[s]).values.astype(int)].astype(np.float32) for s in splits}
assert list(Xp.columns) == mA["columns"], "person-period columns differ from the exported member A"
mB = json.load(open(E / "B/meta.json")); assert list(Xc.columns) == mB["columns"], "clean columns differ from the exported member B"
XB = {s: Xc.reindex(ids[s]).values.astype(np.float32) for s in splits}
mC = json.load(open(E / "C/meta.json")); G = pd.read_parquet(AN / "clean/genomics_features.parquet").join(pd.read_parquet(AN / "clean/cost_features_v2.parquet"), how="left")
assert all(c in Xc.columns for c in mC["columns"]), "member C columns missing from the augmented clean matrix"
XC = {s: Xc.reindex(ids[s])[mC["columns"]].values.astype(float) for s in splits}
# ---- member predictions (fold-averaged for A and B, fit-on-all for C)
pred = {k: {s: np.zeros((len(ids[s]), 40)) for s in splits} for k in names}
kA, kB, kC = [k for k in names if k.startswith("ppds")][0], [k for k in names if k.startswith("screen")][0], [k for k in names if k.startswith("gstack")][0]
for j, c in enumerate(codes):
    for f in range(5):
        pa, pb = E / "A" / f"{c}_fold{f}.json", E / "B" / f"{c}_fold{f}.txt"
        if pa.exists():
            b = xgb.Booster(); b.load_model(str(pa)); cols = np.load(E / "A" / f"{c}_fold{f}_cols.npy")
            for s in splits: pred[kA][s][:, j] += b.predict(xgb.DMatrix(XA[s][:, cols])) / 5
        if pb.exists():
            b = lgb.Booster(model_file=str(pb)); cols = np.load(E / "B" / f"{c}_fold{f}_cols.npy")
            for s in splits: pred[kB][s][:, j] += b.predict(XB[s][:, cols]) / 5
    pc = E / "C" / f"{c}.txt"
    if pc.exists():
        b = lgb.Booster(model_file=str(pc))
        for s in splits: pred[kC][s][:, j] = b.predict(XC[s])
# ---- assembly: zero prior-positive, map through the train-OOF CDF of each member (rank averaging), weight, group-average
oof = {k: np.load(E / "assembly" / f"{k.replace(':', '__')}_oof.npy") for k in names}
cdf = lambda col, x: np.searchsorted(np.sort(col), x, side="right") / len(col)
out = {s: np.zeros((len(ids[s]), 40)) for s in splits}
for j, c in enumerate(codes):
    w = W.loc[int(c) if W.index.dtype != object else c, names].values if (int(c) if W.index.dtype != object else c) in W.index else W.iloc[j][names].values
    for s in splits:
        cols = []
        for k in names:
            tr = np.where(Pv["train"][:, j], 0, oof[k][:, j]); x = np.where(Pv[s][:, j], 0, pred[k][s][:, j])
            cols.append(np.where(Pv[s][:, j], 0, cdf(tr, x) if meta["rank"] else x))
        out[s][:, j] = np.column_stack(cols) @ w
for s in splits:
    out[s] = np.where(Pv[s], 0, out[s])
    for g in meta["groups"]:
        jj = [codes.index(x) for x in g]; m = out[s][:, jj].mean(1, keepdims=True); out[s][:, jj] = np.where(Pv[s][:, jj], 0, m)
if "test" in splits:
    rep = pd.DataFrame(out["test"], index=pd.Index(ids["test"], name="patient_id"), columns=codes); rep.to_csv(ROOT / "predictions_reproduced.csv")
    if (ROOT / "predictions.csv").exists():
        sub = pd.read_csv(ROOT / "predictions.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(ids["test"])[codes].values
        print(f"reproduced vs submitted predictions.csv: max |diff| = {np.abs(sub - out['test']).max():.2e}, mean |diff| = {np.abs(sub - out['test']).mean():.2e}")
if "val" in splits:
    Yv = d["Y"].loc[ids["val"]].values; au = np.mean([roc_auc_score(Yv[:, j], out["val"][:, j]) for j in range(40)]); mp = np.mean([average_precision_score(Yv[:, j], out["val"][:, j]) for j in range(40)])
    print(f"validation split (365 patients, all-patients convention): macro AUROC {au:.4f}  mAP {mp:.4f}")
