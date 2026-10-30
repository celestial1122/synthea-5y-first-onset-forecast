"""Cost features v2 = v1 + pre-anchor spending windows (last 1y / 5y) + two-stage residuals: cross-fitted LightGBM
regressors predict log lifetime cost fields from pre-anchor history (+ cross-fitted logit P(death)); the residual is the
part of lifetime spending that pre-anchor history cannot explain, i.e. window treatment. Writes clean/cost_features_v2.parquet."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "research_intern_coding_test"; HERE = ROOT / "analysis"
C = pd.read_parquet(HERE / "clean/cost_features.parquet"); X = pd.read_parquet(HERE / "clean/X.parquet").reindex(C.index)
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"}).set_index("PATIENT").split.reindex(C.index)
# ---- pre-anchor spending windows
enc_tv = pd.read_csv(DATA / "train_val/encounters.csv", dtype=str, usecols=["PATIENT", "START", "STOP"])
end = enc_tv.STOP.where(enc_tv.STOP.notna() & (enc_tv.STOP != ""), enc_tv.START); end = pd.to_datetime(end, utc=True).dt.tz_localize(None)
anchor = (end.groupby(enc_tv.PATIENT).max() - pd.DateOffset(years=5)); ta = pd.read_csv(DATA / "test_anchors.csv", dtype=str)
anchor = pd.concat([anchor, pd.Series(pd.to_datetime(ta.iloc[:, 1], utc=True).dt.tz_localize(None).values, index=ta.iloc[:, 0].values)])
parts = []
for s in ("train_val", "test"):
    for name, dcol, ccol in (("encounters", "START", "TOTAL_CLAIM_COST"), ("medications", "START", "TOTALCOST"), ("procedures", "DATE", "BASE_COST"), ("immunizations", "DATE", "BASE_COST")):
        d = pd.read_csv(DATA / s / f"{name}.csv", dtype=str, usecols=["PATIENT", dcol, ccol]); t = pd.to_datetime(d[dcol], utc=True, errors="coerce").dt.tz_localize(None)
        a = d.PATIENT.map(anchor); yrs = (a - t).dt.days / 365.25; c = d[ccol].astype(float)
        parts.append(pd.DataFrame({"PATIENT": d.PATIENT, "c1": np.where((yrs > 0) & (yrs <= 1), c, 0), "c5": np.where((yrs > 0) & (yrs <= 5), c, 0), "n5": ((yrs > 0) & (yrs <= 5)).astype(int) if name == "encounters" else 0}))
W = pd.concat(parts).groupby("PATIENT").sum().reindex(C.index).fillna(0)
V = C.copy(); V["cost_pre_1y"] = W.c1; V["cost_pre_5y"] = W.c5; V["cost_pre_5y_per_enc"] = W.c5 / W.n5.clip(lower=1); V["cost_pre_5y_share"] = W.c5 / C.cost_pre_total.replace(0, np.nan)
# ---- two-stage residuals (train-only fit, cross-fitted on train)
pd_ = {}
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id", "DEATHDATE"]).rename(columns={"Id": "PATIENT"}).set_index("PATIENT")
for s, k in (("train", "oof"), ("val", "val"), ("test", "test")):
    ids_s = (pats.index if s != "test" else pd.Index(pd.read_csv(DATA / "test/patients.csv", dtype=str).Id)); ids_s = ids_s[spl.reindex(ids_s).values == s]
    arr = np.load(HERE / "stack" / f"pdeath_{k}.npy"); assert len(arr) == len(ids_s), (s, len(arr), len(ids_s)); pd_[s] = pd.Series(arr, index=ids_s)
pdeath = pd.concat(pd_.values()).reindex(C.index); lp = np.log(np.clip(pdeath, 1e-4, 1 - 1e-4) / (1 - np.clip(pdeath, 1e-4, 1 - 1e-4)))
F = np.column_stack([X.values.astype(np.float32), V[["cost_pre_total", "cost_pre_cov", "cost_pre_1y", "cost_pre_5y", "cost_pre_per_year"]].values, lp.values])
tr = (spl.values == "train"); params = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_child_samples=20, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6, seed=0)
for col in ("cost_life_cov", "cost_life_exp", "cost_life_total"):
    y = np.log1p(V[col].clip(lower=0).values); pred = np.zeros(len(V)); itr = np.where(tr)[0]
    for a, b in KFold(5, shuffle=True, random_state=0).split(itr):
        m = lgb.train(params, lgb.Dataset(F[itr[a]], y[itr[a]]), 400); pred[itr[b]] = m.predict(F[itr[b]])
    m = lgb.train(params, lgb.Dataset(F[itr], y[itr]), 400); pred[~tr] = m.predict(F[~tr])
    V[col.replace("life", "pred")] = np.expm1(pred); V[col.replace("life", "resid")] = y - pred
    r2 = 1 - np.var(y[tr] - pred[tr]) / np.var(y[tr]); print(f"{col}: cross-fitted R^2 on train {r2:.3f}")
V["cost_resid_cov_x_alive"] = V.cost_resid_cov * (1 - pdeath.values); V["cost_resid_cov_x_dead"] = V.cost_resid_cov * pdeath.values
V.to_parquet(HERE / "clean/cost_features_v2.parquet"); print("saved", V.shape)
# ---- diagnostics: raw vs residual, train at-risk
pats2 = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"}); ids = pats2.merge(spl.reset_index(), on="PATIENT").query("split=='train'").PATIENT.values
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0) & (M["split"] == "train"); pos = pd.Series(np.arange(off.sum()), index=M["pid"][off].astype(str)).reindex(ids).values.astype(int)
Y = M["y"][off][pos]; P = M["prior"][off][pos].astype(bool); codes = list(M["codes"]); Vt = V.reindex(ids); died = pats.DEATHDATE.reindex(ids).notna().values
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str); rows = []
for j, c in enumerate(codes):
    at = ~P[:, j]; y = Y[at, j]; d = died[at]
    r = {"cond": o.loc[c, "cond"][:24], "pos": int(y.sum())}
    for k in ("cost_life_cov", "cost_post_cov", "cost_resid_cov", "cost_resid_total", "cost_pre_5y"): a = auc(y, Vt[k].values[at]); r[k] = round(max(a, 1 - a), 3)
    r["resid_died"] = round(auc(y[d], Vt.cost_resid_cov.values[at][d]), 3) if 0 < y[d].sum() < d.sum() else np.nan
    r["resid_alive"] = round(auc(y[~d], Vt.cost_resid_cov.values[at][~d]), 3) if 0 < y[~d].sum() < (~d).sum() else np.nan
    rows.append(r)
R = pd.DataFrame(rows, index=codes); pd.set_option("display.width", 220); print(R.sort_values("cost_resid_cov", ascending=False).to_string()); print(f"[{time.time()-t0:.0f}s]")
