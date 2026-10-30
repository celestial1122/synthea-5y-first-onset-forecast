"""Gate-aware combiner (member D of the simplified pipeline): per target, a small LightGBM on the three members' scores
plus the cross-fitted logit P(death) and age, so the death-gate x cause structure is modelled explicitly instead of
being left to rank averaging. Inputs are the members' honest OOF predictions for train rows; 5-fold patient CV gives an
honest OOF for D; fit-on-all gives val/test. Usage: gate_combiner.py <out_stem> <dir:stem> <dir:stem> <dir:stem>
Outputs gstack/<out_stem>_{oof,val,test}.npy and prints D alone vs the best single member."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"; OUT = HERE / "gstack"
LR = "--lr" in sys.argv; args_ = [a for a in sys.argv[1:] if a != "--lr"]; out_stem, members = args_[0], args_[1:]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"}); spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ps = pats.merge(spl, on="PATIENT"); vp = pd.read_csv(HERE / "final/val_predictions_final.csv", dtype={0: str}); tp = pd.read_csv(ROOT / "predictions.csv", dtype={0: str})
ids = {"train": ps.query("split=='train'").PATIENT.values, "val": vp.iloc[:, 0].values, "test": tp.iloc[:, 0].values}
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0); codes = list(M["codes"])
def lab(split, order):
    m = off & (M["split"] == split); pos = pd.Series(np.arange(m.sum()), index=M["pid"][m].astype(str)).reindex(order).values.astype(int)
    return M["y"][m][pos], M["prior"][m][pos].astype(bool)
Y, P = {}, {}
for s in ids: Y[s], P[s] = lab(s, ids[s])
S = {s: [np.load(HERE / m.split(":")[0] / f"{m.split(':')[1]}_{k}.npy") for m in members] for s, k in (("train", "oof"), ("val", "val"), ("test", "test"))}
pdv = {"train": np.load(HERE / "stack/pdeath_oof.npy"), "val": np.load(HERE / "stack/pdeath_val.npy"), "test": np.load(HERE / "stack/pdeath_test.npy")}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
X = pd.read_parquet(HERE / "clean/X.parquet"); age = {s: X["age"].reindex(ids[s]).values.astype(float) for s in ids}
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str); grp = o.grp.reindex(codes).values
def feats(s, j, rows=None):
    lgs = [lg(a[:, j]) for a in S[s]]; lp = lg(pdv[s]); cols = lgs + [lp, age[s]]
    if LR: cols += [l * lp for l in lgs]          # explicit gate x member interactions for the linear combiner
    F = np.column_stack(cols); return F if rows is None else F[rows]
def fit_pred(Xa, ya, Xs):
    if LR:
        sc = StandardScaler().fit(Xa); m = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xa), ya); return [m.predict_proba(sc.transform(x))[:, 1] for x in Xs]
    m = lgb.train(params, lgb.Dataset(Xa, ya), 250); return [m.predict(x) for x in Xs]
params = dict(objective="binary", learning_rate=0.03, num_leaves=4, min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=0)
oof = np.zeros((len(ids["train"]), 40)); pv = np.zeros((len(ids["val"]), 40)); pt = np.zeros((len(ids["test"]), 40))
for j in range(40):
    at = np.where(~P["train"][:, j])[0]; y = Y["train"][at, j]; Xj = feats("train", j, at); oo = np.zeros(len(at))
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xj):
        oo[b] = fit_pred(Xj[a], y[a], [Xj[b]])[0]
    pv[:, j], pt[:, j] = fit_pred(Xj, y, [feats("val", j), feats("test", j)]); oof[at, j] = oo
def mAP(Yv, Pv, A): return np.array([ap(Yv[~Pv[:, j], j], A[~Pv[:, j], j]) for j in range(40)])
d_tr, d_va = mAP(Y["train"], P["train"], oof), mAP(Y["val"], P["val"], pv)
best = max(members, key=lambda m: mAP(Y["train"], P["train"], S["train"][members.index(m)]).mean()); b_tr = mAP(Y["train"], P["train"], S["train"][members.index(best)])
print(f"RESULT {out_stem}: OOF mAP={d_tr.mean():.4f} | acc {d_tr[grp=='acc'].mean():.3f} / term {d_tr[grp=='term'].mean():.3f} / chain {d_tr[grp=='chain'].mean():.3f} | val {d_va.mean():.4f} | best single member {best} OOF {b_tr.mean():.4f} [{time.time()-t0:.0f}s]")
for s, arr in (("oof", oof), ("val", pv), ("test", pt)): np.save(OUT / f"{out_stem}_{s}.npy", arr)
