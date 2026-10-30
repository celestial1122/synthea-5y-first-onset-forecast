"""Is the official-row LightGBM under-trained? Train 1200 rounds per (target, fold) and evaluate the OOF mAP at several
numbers of boosting rounds with the same config as screen_gbm/pipeline_clean (lr 0.03, 8 leaves). Features = clean X
+ genomics + cost v3 (EHR_GENOMICS=4)."""
import os; os.environ["EHR_GENOMICS"] = "4"
import numpy as np, pandas as pd, lightgbm as lgb, time, genfeat
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"}); spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ids = pats.merge(spl, on="PATIENT").query("split=='train'").PATIENT.values
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0) & (M["split"] == "train"); pos = pd.Series(np.arange(off.sum()), index=M["pid"][off].astype(str)).reindex(ids).values.astype(int)
Y = M["y"][off][pos]; P = M["prior"][off][pos].astype(bool); codes = list(M["codes"])
X = genfeat.augment(pd.read_parquet(HERE / "clean/X.parquet")).reindex(ids).values.astype(np.float32)
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str); grp = o.grp.reindex(codes).values
KS = [50, 100, 200, 350, 500, 800, 1200]
cfg = dict(objective="binary", verbose=-1, n_jobs=4, learning_rate=0.03, n_estimators=1200, subsample=0.8, subsample_freq=1, reg_lambda=5.0, num_leaves=8, min_child_samples=15, colsample_bytree=0.3, random_state=0)
oof = {k: np.zeros((len(ids), 40)) for k in KS}
for j in range(40):
    at = ~P[:, j]
    for a, b in KFold(5, shuffle=True, random_state=0).split(ids):
        ma = a[at[a]]; m = lgb.LGBMClassifier(**cfg).fit(X[ma], Y[ma, j])
        for k in KS: oof[k][b, j] = m.predict_proba(X[b], num_iteration=k)[:, 1]
    if j % 8 == 7: print(f"  target {j+1}/40 [{time.time()-t0:.0f}s]", flush=True)
rows = []
for k in KS:
    a = np.array([ap(Y[~P[:, j], j], oof[k][~P[:, j], j]) for j in range(40)])
    rows.append({"rounds": k, "mAP": a.mean(), **{g: a[grp == g].mean() for g in ("acc", "chain", "term")}})
R = pd.DataFrame(rows).set_index("rounds"); print(R.round(4).to_string())
per = pd.DataFrame({k: [ap(Y[~P[:, j], j], oof[k][~P[:, j], j]) for j in range(40)] for k in KS}, index=[o.cond.reindex(codes).str[:24].values]); per["best_k"] = per.idxmax(1)
print("\nbest rounds per target (counts):", per.best_k.value_counts().to_dict()); print(f"[{time.time()-t0:.0f}s]")
R.to_csv(HERE / "oracle/rounds_curve_lgb_gcr3.csv")
