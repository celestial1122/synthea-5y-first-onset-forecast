"""Quick genomics-aware second stage: per target, LightGBM on [final assembly score, genomics/imaging features].
Train rows = official train patients at risk; 5-fold patient CV gives an honest OOF; fit-on-all gives val/test.
Outputs gstack/final+gen_{oof,val,test}.npy (candidate layout) and prints base vs stacked mAP."""
import numpy as np, pandas as pd, lightgbm as lgb, json, time
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"; OUT = HERE / "gstack"; OUT.mkdir(exist_ok=True)
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"})
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ps = pats.merge(spl, on="PATIENT", how="inner"); ids = {"train": ps.query("split=='train'").PATIENT.values}
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); codes = list(M["codes"]); off = (M["k"] == 0)
def lab(split, order):
    m = off & (M["split"] == split); pos = pd.Series(np.arange(m.sum()), index=M["pid"][m].astype(str)).reindex(order).values
    assert not np.isnan(pos.astype(float)).any(); return M["y"][m][pos.astype(int)], M["prior"][m][pos.astype(int)].astype(bool)
vp = pd.read_csv(HERE / "final/val_predictions_final.csv", dtype={0: str}); tp = pd.read_csv(ROOT / "predictions.csv", dtype={0: str})
idc = vp.columns[0]; ids["val"] = vp[idc].values; ids["test"] = tp[idc].values
S = {"train": np.load(HERE / "final/oof_crossfit.npy"), "val": vp[codes].values.astype(float), "test": tp[codes].values.astype(float)}
Y, P = {}, {}
Y["train"], P["train"] = lab("train", ids["train"]); Y["val"], P["val"] = lab("val", ids["val"])
# test prior from meta too (needed to zero prior-positive rows)
_, P["test"] = lab("test", ids["test"]) if (off & (M["split"] == "test")).sum() else (None, np.zeros((len(ids["test"]), 40), bool))
import sys; COST = any(a in sys.argv[1:] for a in ("cost", "cost2", "cost3")); COST2 = "cost2" in sys.argv[1:]; COST3 = "cost3" in sys.argv[1:]; TAG = "final8+genc3" if COST3 else "final7+genc2" if COST2 else ("final7+genc" if COST else "final+gen")
G = pd.read_parquet(HERE / "clean/genomics_features.parquet")
if COST: G = G.join(pd.read_parquet(HERE / ("clean/cost_features_v3.parquet" if COST3 else "clean/cost_features_v2.parquet" if COST2 else "clean/cost_features.parquet")), how="left")
gcols = list(G.columns)
F = {s: np.column_stack([S[s], G.reindex(ids[s]).values.astype(float)]) for s in S}   # [40 scores | gen feats]
def mAP(Yv, Pv, A): return np.array([ap(Yv[:, j], np.where(Pv[:, j], 0, A[:, j])) for j in range(40)])
base_tr, base_va = mAP(Y["train"], P["train"], S["train"]), mAP(Y["val"], P["val"], S["val"])
params = dict(objective="binary", learning_rate=0.03, num_leaves=4, min_child_samples=15, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=0)
oof = S["train"].copy(); pv = S["val"].copy(); pt = S["test"].copy(); rows = []
for j, c in enumerate(codes):
    at = ~P["train"][:, j]; y = Y["train"][at, j]
    def feats(s, m=None):
        A = F[s] if m is None else F[s][m]; return np.column_stack([A[:, j], A[:, 40:]])
    Xj = feats("train", at); o = np.zeros(at.sum())
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xj):
        b = lgb.train(params, lgb.Dataset(Xj[tr], y[tr]), 400); o[te] = b.predict(Xj[te])
    b = lgb.train(params, lgb.Dataset(Xj, y), 400)
    oof[at, j] = o; pv[:, j] = b.predict(feats("val")); pt[:, j] = b.predict(feats("test"))
    imp = pd.Series(b.feature_importance("gain"), index=["score"] + gcols); top = imp.sort_values(ascending=False).head(3)
    rows.append({"code": c, "cond": M["codes"][j], "base_oof": base_tr[j], "stack_oof": ap(y, o), "n_pos": int(y.sum()),
                 "top_feats": "; ".join(f"{k}:{v/imp.sum():.2f}" for k, v in top.items())})
R = pd.DataFrame(rows); R["gain"] = R.stack_oof - R.base_oof
st_tr, st_va = mAP(Y["train"], P["train"], oof), mAP(Y["val"], P["val"], pv)
R["base_val"] = base_va; R["stack_val"] = st_va
sel = R.gain.values > 0.01
oof_s = np.where(sel[None, :], oof, S["train"]); pv_s = np.where(sel[None, :], pv, S["val"]); pt_s = np.where(sel[None, :], pt, S["test"])
se_tr, se_va = mAP(Y["train"], P["train"], oof_s), mAP(Y["val"], P["val"], pv_s)
def auroc(Yv, Pv, A): return np.mean([auc(Yv[~Pv[:, j], j], A[~Pv[:, j], j]) for j in range(40)])
pd.set_option("display.width", 250)
print(R.sort_values("gain", ascending=False)[["cond", "n_pos", "base_oof", "stack_oof", "gain", "base_val", "stack_val", "top_feats"]].round(3).to_string(index=False))
print(f"\nbase       : OOF {base_tr.mean():.4f}  val {base_va.mean():.4f}  val AUROC {auroc(Y['val'], P['val'], S['val']):.4f}")
print(f"stack all  : OOF {st_tr.mean():.4f}  val {st_va.mean():.4f}  val AUROC {auroc(Y['val'], P['val'], pv):.4f}")
print(f"stack sel  : OOF {se_tr.mean():.4f}  val {se_va.mean():.4f}  val AUROC {auroc(Y['val'], P['val'], pv_s):.4f}   selected {int(sel.sum())} targets: {[M['codes'][j][:18] for j in np.where(sel)[0]]}")
for name, a, b_, c_ in ((TAG, oof, pv, pt), (TAG + "_sel", oof_s, pv_s, pt_s)):
    np.save(OUT / f"{name}_oof.npy", a); np.save(OUT / f"{name}_val.npy", b_); np.save(OUT / f"{name}_test.npy", c_)
R.to_csv(OUT / f"per_condition_{TAG}.csv", index=False)
json.dump({"base": [base_tr.mean(), base_va.mean()], "stack_all": [st_tr.mean(), st_va.mean()], "stack_sel": [se_tr.mean(), se_va.mean()], "selected": [codes[j] for j in np.where(sel)[0]]}, open(OUT / f"summary_{TAG}.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s]")
