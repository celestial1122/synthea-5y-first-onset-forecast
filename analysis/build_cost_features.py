"""Lifetime cost fields (patients.csv HEALTHCARE_EXPENSES / HEALTHCARE_COVERAGE) decomposed against the pre-anchor
cost that can be summed from the event tables, so that the model sees 'post-anchor spending' explicitly instead of a
lifetime total. Writes analysis/clean/cost_features.parquet (index PATIENT). Also prints the accounting identity check
on full train/val records and univariate diagnostics on train."""
import numpy as np, pandas as pd, time
from pathlib import Path
from sklearn.metrics import roc_auc_score as auc
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "research_intern_coding_test"; HERE = ROOT / "analysis"
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"}).set_index("PATIENT").split
P = pd.concat([pd.read_csv(DATA / s / "patients.csv", dtype=str) for s in ("train_val", "test")]).rename(columns={"Id": "PATIENT"}).set_index("PATIENT")
P = P[P.index.isin(spl.index)]
life_exp = P.HEALTHCARE_EXPENSES.astype(float); life_cov = P.HEALTHCARE_COVERAGE.astype(float)
# anchors
enc_tv = pd.read_csv(DATA / "train_val/encounters.csv", dtype=str, usecols=["PATIENT", "START", "STOP", "TOTAL_CLAIM_COST", "PAYER_COVERAGE", "BASE_ENCOUNTER_COST"])
end = enc_tv.STOP.where(enc_tv.STOP.notna() & (enc_tv.STOP != ""), enc_tv.START); end = pd.to_datetime(end, utc=True).dt.tz_localize(None)
anchor = (end.groupby(enc_tv.PATIENT).max() - pd.DateOffset(years=5))
ta = pd.read_csv(DATA / "test_anchors.csv", dtype=str); anchor = pd.concat([anchor, pd.Series(pd.to_datetime(ta.iloc[:, 1], utc=True).dt.tz_localize(None).values, index=ta.iloc[:, 0].values)])
def load(s, name, datecol, cols):
    d = pd.read_csv(DATA / s / f"{name}.csv", dtype=str, usecols=["PATIENT", datecol] + cols)
    d["t"] = pd.to_datetime(d[datecol], utc=True, errors="coerce").dt.tz_localize(None)
    for c in cols: d[c] = d[c].astype(float)
    return d
tabs = {}
for s in ("train_val", "test"):
    tabs[s] = {"enc": load(s, "encounters", "START", ["TOTAL_CLAIM_COST", "PAYER_COVERAGE", "BASE_ENCOUNTER_COST"]),
               "med": load(s, "medications", "START", ["TOTALCOST", "PAYER_COVERAGE", "BASE_COST"]),
               "proc": load(s, "procedures", "DATE", ["BASE_COST"]), "imm": load(s, "immunizations", "DATE", ["BASE_COST"])}
def sums(t, mask=None):
    g = lambda d, c: (d[c] if mask is None else d[c][mask[id(d)]]).groupby(d.PATIENT if mask is None else d.PATIENT[mask[id(d)]]).sum()
    return pd.DataFrame({"enc_total": g(t["enc"], "TOTAL_CLAIM_COST"), "enc_cov": g(t["enc"], "PAYER_COVERAGE"), "enc_base": g(t["enc"], "BASE_ENCOUNTER_COST"),
                         "med_total": g(t["med"], "TOTALCOST"), "med_cov": g(t["med"], "PAYER_COVERAGE"), "proc": g(t["proc"], "BASE_COST"), "imm": g(t["imm"], "BASE_COST")}).fillna(0)
# identity check on FULL train/val records
full = sums(tabs["train_val"]).reindex(life_exp.index.intersection(sums(tabs["train_val"]).index)).fillna(0)
lt = (life_exp + life_cov).reindex(full.index); lc = life_cov.reindex(full.index)
for name, comb in {"enc_total": full.enc_total, "enc_total+med_total": full.enc_total + full.med_total, "enc_total+med+proc+imm": full.enc_total + full.med_total + full.proc + full.imm,
                   "enc_base+med+proc+imm": full.enc_base + full.med_total + full.proc + full.imm}.items():
    r = (comb / lt.replace(0, np.nan)); print(f"identity total: {name:26s} corr {np.corrcoef(comb, lt)[0,1]:.4f}  median ratio {r.median():.3f}  IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f}")
for name, comb in {"enc_cov": full.enc_cov, "enc_cov+med_cov": full.enc_cov + full.med_cov}.items():
    r = comb / lc.replace(0, np.nan); print(f"identity coverage: {name:22s} corr {np.corrcoef(comb, lc)[0,1]:.4f}  median ratio {r.median():.3f}")
# pre-anchor sums for every patient (train/val filtered at own anchor; test tables already truncated, filter anyway)
pre = []
for s in ("train_val", "test"):
    t = tabs[s]; mask = {}
    for k, d in t.items(): mask[id(d)] = (d.t < d.PATIENT.map(anchor)).values
    pre.append(sums(t, mask))
pre = pd.concat(pre); pre = pre[~pre.index.duplicated()].reindex(life_exp.index).fillna(0)
pre_total = pre.enc_total + pre.med_total + pre.proc + pre.imm; pre_cov = pre.enc_cov + pre.med_cov
F = pd.DataFrame(index=life_exp.index)
F["cost_life_exp"] = life_exp; F["cost_life_cov"] = life_cov; F["cost_life_total"] = life_exp + life_cov
F["cost_pre_total"] = pre_total; F["cost_pre_cov"] = pre_cov; F["cost_pre_exp"] = pre_total - pre_cov
F["cost_post_total"] = F.cost_life_total - pre_total; F["cost_post_cov"] = life_cov - pre_cov; F["cost_post_exp"] = life_exp - F.cost_pre_exp
F["cost_post_share"] = F.cost_post_total / F.cost_life_total.replace(0, np.nan)
age = pd.read_parquet(HERE / "clean/X.parquet")["age"].reindex(F.index).astype(float)
F["cost_pre_per_year"] = pre_total / age.clip(lower=1); F["cost_post_per_year"] = F.cost_post_total / 5.0; F["cost_post_vs_pre_rate"] = F.cost_post_per_year / (F.cost_pre_per_year + 1)
F.index.name = "PATIENT"; F.to_parquet(HERE / "clean/cost_features.parquet")
print(f"saved {F.shape}; post_total quantiles train:", F.cost_post_total[spl.reindex(F.index) == "train"].quantile([.1, .5, .9]).round(0).tolist(), "test:", F.cost_post_total[spl.reindex(F.index) == "test"].quantile([.1, .5, .9]).round(0).tolist())
# diagnostics on train at-risk patients
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"})
ids = pats.merge(spl.reset_index(), on="PATIENT").query("split=='train'").PATIENT.values
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0) & (M["split"] == "train")
pos = pd.Series(np.arange(off.sum()), index=M["pid"][off].astype(str)).reindex(ids).values.astype(int); Y = M["y"][off][pos]; Pr = M["prior"][off][pos].astype(bool); codes = list(M["codes"])
died = P.DEATHDATE.reindex(ids).notna().values
Ft = F.reindex(ids)
print("\nAUROC vs death-in-window: life_total %.3f  post_total %.3f  post_share %.3f  post_per_year %.3f" % tuple(auc(died, Ft[c].fillna(0)) for c in ("cost_life_total", "cost_post_total", "cost_post_share", "cost_post_per_year")))
rows = []
for j, c in enumerate(codes):
    at = ~Pr[:, j]; y = Y[at, j]
    rows.append({"code": c, "pos": int(y.sum()), **{k: auc(y, Ft[k].values[at]) for k in ("cost_life_total", "cost_life_cov", "cost_post_total", "cost_post_cov", "cost_post_share", "cost_post_vs_pre_rate")}})
R = pd.DataFrame(rows).set_index("code"); R["best"] = R.drop(columns="pos").max(1)
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str); R.insert(0, "cond", o.cond.reindex(R.index).str[:26])
pd.set_option("display.width", 220); print(R.sort_values("best", ascending=False).round(3).to_string()); print(f"[{time.time()-t0:.0f}s]")
