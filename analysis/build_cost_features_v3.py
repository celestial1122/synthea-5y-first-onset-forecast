"""Cost features v3 = v2 + pre-anchor payer mix (share of pre-anchor encounters and medications per payer id, n payers,
dominant-payer share). Payer identity determines premiums / coverage ratios in Synthea, so it should explain more of
the lifetime totals and sharpen the two-stage residual. Writes clean/cost_features_v3.parquet and prints R^2 / AUROC
comparison against v2."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "research_intern_coding_test"; HERE = ROOT / "analysis"
V2 = pd.read_parquet(HERE / "clean/cost_features_v2.parquet"); X = pd.read_parquet(HERE / "clean/X.parquet").reindex(V2.index)
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"}).set_index("PATIENT").split.reindex(V2.index)
enc_tv = pd.read_csv(DATA / "train_val/encounters.csv", dtype=str, usecols=["PATIENT", "START", "STOP"])
end = enc_tv.STOP.where(enc_tv.STOP.notna() & (enc_tv.STOP != ""), enc_tv.START); end = pd.to_datetime(end, utc=True).dt.tz_localize(None)
anchor = (end.groupby(enc_tv.PATIENT).max() - pd.DateOffset(years=5)); ta = pd.read_csv(DATA / "test_anchors.csv", dtype=str)
anchor = pd.concat([anchor, pd.Series(pd.to_datetime(ta.iloc[:, 1], utc=True).dt.tz_localize(None).values, index=ta.iloc[:, 0].values)])
parts = []
for s in ("train_val", "test"):
    for name, dcol in (("encounters", "START"), ("medications", "START")):
        d = pd.read_csv(DATA / s / f"{name}.csv", dtype=str, usecols=["PATIENT", dcol, "PAYER"]); t = pd.to_datetime(d[dcol], utc=True, errors="coerce").dt.tz_localize(None)
        d = d[t < d.PATIENT.map(anchor)]; parts.append(d[["PATIENT", "PAYER"]].assign(src=name))
E = pd.concat(parts); payers = sorted(E.PAYER.dropna().unique()); print("payers", len(payers))
ct = pd.crosstab(E.PATIENT, E.PAYER).reindex(V2.index).fillna(0); share = ct.div(ct.sum(1).replace(0, np.nan), axis=0)
share.columns = [f"payer_share_{i}" for i in range(len(share.columns))]
V3 = V2.join(share); V3["payer_n"] = (ct > 0).sum(1); V3["payer_dominant_share"] = share.max(1)
# last payer before anchor (most recent encounter's payer)
encs = []
for s in ("train_val", "test"):
    d = pd.read_csv(DATA / s / "encounters.csv", dtype=str, usecols=["PATIENT", "START", "PAYER"]); d["t"] = pd.to_datetime(d.START, utc=True, errors="coerce").dt.tz_localize(None)
    d = d[d.t < d.PATIENT.map(anchor)]; encs.append(d.sort_values("t").groupby("PATIENT").PAYER.last())
lastp = pd.concat(encs); lastp = lastp[~lastp.index.duplicated()].reindex(V3.index)
for i, pid in enumerate(payers): V3[f"payer_last_{i}"] = (lastp == pid).astype(int)
# residual model with payer features
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id", "DEATHDATE"]).rename(columns={"Id": "PATIENT"}).set_index("PATIENT")
pdv = {}
for s, k in (("train", "oof"), ("val", "val"), ("test", "test")):
    ids_s = (pats.index if s != "test" else pd.Index(pd.read_csv(DATA / "test/patients.csv", dtype=str).Id)); ids_s = ids_s[spl.reindex(ids_s).values == s]
    pdv[s] = pd.Series(np.load(HERE / "stack" / f"pdeath_{k}.npy"), index=ids_s)
pdeath = pd.concat(pdv.values()).reindex(V3.index); lp = np.log(np.clip(pdeath, 1e-4, 1 - 1e-4) / (1 - np.clip(pdeath, 1e-4, 1 - 1e-4)))
pay_cols = [c for c in V3.columns if c.startswith("payer_")]
F = np.column_stack([X.values.astype(np.float32), V3[["cost_pre_total", "cost_pre_cov", "cost_pre_1y", "cost_pre_5y", "cost_pre_per_year"] + pay_cols].values, lp.values])
tr = (spl.values == "train"); itr = np.where(tr)[0]
params = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_child_samples=20, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6, seed=0)
for col in ("cost_life_cov", "cost_life_exp", "cost_life_total"):
    y = np.log1p(V3[col].clip(lower=0).values); pred = np.zeros(len(V3))
    for a, b in KFold(5, shuffle=True, random_state=0).split(itr):
        m = lgb.train(params, lgb.Dataset(F[itr[a]], y[itr[a]]), 400); pred[itr[b]] = m.predict(F[itr[b]])
    m = lgb.train(params, lgb.Dataset(F[itr], y[itr]), 400); pred[~tr] = m.predict(F[~tr])
    V3[col.replace("life", "pred3")] = np.expm1(pred); V3[col.replace("life", "resid3")] = y - pred
    r2 = 1 - np.var(y[tr] - pred[tr]) / np.var(y[tr]); r2_v2 = 1 - np.var(np.log1p(V2[col].clip(lower=0).values[tr]) - np.log1p(V2[col.replace("life", "pred")].values[tr])) / np.var(y[tr])
    print(f"{col}: R^2 v2 {r2_v2:.3f} -> v3 (+payer) {r2:.3f}")
V3["cost_resid3_cov_x_alive"] = V3.cost_resid3_cov * (1 - pdeath.values); V3["cost_resid3_cov_x_dead"] = V3.cost_resid3_cov * pdeath.values
V3.to_parquet(HERE / "clean/cost_features_v3.parquet"); print("saved", V3.shape)
# diagnostics
pats2 = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"}); ids = pats2.merge(spl.reset_index(), on="PATIENT").query("split=='train'").PATIENT.values
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0) & (M["split"] == "train"); pos = pd.Series(np.arange(off.sum()), index=M["pid"][off].astype(str)).reindex(ids).values.astype(int)
Y = M["y"][off][pos]; P = M["prior"][off][pos].astype(bool); codes = list(M["codes"]); Vt = V3.reindex(ids)
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str); rows = []
for j, c in enumerate(codes):
    at = ~P[:, j]; y = Y[at, j]; r = {"cond": o.loc[c, "cond"][:24], "pos": int(y.sum())}
    for k in ("cost_resid_cov", "cost_resid3_cov", "cost_resid_total", "cost_resid3_total", "payer_dominant_share"): a = auc(y, Vt[k].values[at]); r[k] = round(max(a, 1 - a), 3)
    rows.append(r)
R = pd.DataFrame(rows, index=codes); pd.set_option("display.width", 200); print(R.sort_values("cost_resid3_cov", ascending=False).head(16).to_string())
print("mean |delta| resid3 - resid (cov):", round((R.cost_resid3_cov - R.cost_resid_cov).mean(), 4), f"[{time.time()-t0:.0f}s]")
