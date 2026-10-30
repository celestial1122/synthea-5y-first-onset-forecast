"""Where does each target's signal live? For every target (train, at-risk rows) the best univariate AUROC within each
feature group: static/demographics, prior-target flags, history tokens (counts/recency/first), numeric observation
summaries, active meds, genomics, lifetime-cost fields; plus the final-assembly AP. Also checks whether the cost signal
is conditional on dying in the window. Output: oracle/feature_group_map.csv"""
import numpy as np, pandas as pd, time
from pathlib import Path
from scipy.stats import rankdata
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id", "DEATHDATE"]).rename(columns={"Id": "PATIENT"})
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ps = pats.merge(spl, on="PATIENT", how="inner").query("split=='train'"); ids = ps.PATIENT.values; died = ps.DEATHDATE.notna().values
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0) & (M["split"] == "train")
pos = pd.Series(np.arange(off.sum()), index=M["pid"][off].astype(str)).reindex(ids).values.astype(int); Y = M["y"][off][pos]; P = M["prior"][off][pos].astype(bool); codes = list(M["codes"])
X = pd.read_parquet(HERE / "clean/X.parquet").reindex(ids); G = pd.read_parquet(HERE / "clean/genomics_features.parquet").reindex(ids); C = pd.read_parquet(HERE / "clean/cost_features.parquet").reindex(ids)
tok = lambda c: c.startswith(("cnt_", "c5_", "c1_", "rec_", "first_")); numr = lambda c: c.startswith(("last_", "max_", "min_", "m5_", "std_", "nobs_", "trend_"))
groups = {"static": [c for c in X.columns if not tok(c) and not numr(c) and not c.startswith(("active_", "prior_"))],
          "prior": [c for c in X.columns if c.startswith("prior_")], "tokens": [c for c in X.columns if tok(c)], "numeric": [c for c in X.columns if numr(c)],
          "active_med": [c for c in X.columns if c.startswith("active_")], "genomics": list(G.columns), "cost": list(C.columns)}
A = pd.concat([X, G, C], axis=1)
def auc_matrix(Xm, y):   # rank-based AUROC for every column at once (NaN -> lowest), max over sign
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.full(Xm.shape[1], np.nan)
    Z = np.where(np.isnan(Xm), -np.inf, Xm); R = np.apply_along_axis(rankdata, 0, Z)
    a = (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0); return np.maximum(a, 1 - a)
fin = np.load(HERE / "final/oof_crossfit.npy"); o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str)
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
rows = []
for j, c in enumerate(codes):
    at = ~P[:, j]; y = Y[at, j]; r = {"code": c, "cond": o.loc[c, "cond"][:26], "grp": o.loc[c, "grp"], "pos": int(y.sum()), "final7_AP": round(ap(y, fin[at, j]), 3)}
    for g, cols in groups.items():
        a = auc_matrix(A.loc[:, cols].values.astype(float)[at], y); k = np.nanargmax(a); r[g] = round(float(a[k]), 3); r[g + "_best"] = cols[k][:28]
    # cost signal conditional on death status
    cv = C.cost_post_cov.values[at]; d = died[at]
    r["cost_auc_died"] = round(auc(y[d], cv[d]), 3) if 0 < y[d].sum() < d.sum() else np.nan
    r["cost_auc_alive"] = round(auc(y[~d], cv[~d]), 3) if 0 < y[~d].sum() < (~d).sum() else np.nan
    r["pos_share_died"] = round(float(y[d].sum() / max(y.sum(), 1)), 2)
    rows.append(r)
R = pd.DataFrame(rows).set_index("code"); R.to_csv(HERE / "oracle/feature_group_map.csv")
pd.set_option("display.width", 250)
print(R[["cond", "grp", "pos", "final7_AP", "static", "prior", "tokens", "numeric", "active_med", "genomics", "cost", "cost_auc_died", "cost_auc_alive", "pos_share_died"]].sort_values(["grp", "final7_AP"]).to_string())
print("\nbest history feature per target (tokens):"); print(R[["cond", "tokens", "tokens_best", "cost", "cost_best", "genomics", "genomics_best"]].sort_values("cost", ascending=False).head(16).to_string())
print(f"[{time.time()-t0:.0f}s]")
