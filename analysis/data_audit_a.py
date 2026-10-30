"""Comprehensive data audit, part A: cohort/splits, covariate shift, tables, time structure, calendar-era markers, numeric observations."""
import warnings, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from ehr_load import load, DATA, _d
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
d = load(); pats, ev, Y, P, codes, targets = d["pats"], d["events"], d["Y"], d["P"], d["codes"], d["targets"]
pidx = pats.set_index("PATIENT"); out = {}
print("#" * 100); print("A1. COHORT BY SPLIT"); print("#" * 100)
pre = ev[ev.pre].copy(); g = pre.groupby("PATIENT")
pats["n_pre_events"] = g.size().reindex(pats.PATIENT).fillna(0).values
pats["hist_years"] = g.years_before.max().reindex(pats.PATIENT).fillna(0).values
pats["first_event_age"] = (pats.age - pats.hist_years)
pats["n_pre_cond"] = pre[pre.kind == "COND"].groupby("PATIENT").size().reindex(pats.PATIENT).fillna(0).values
rows = []
for s in ("train", "val", "test"):
    p = pats[pats.split == s]
    rows.append(dict(split=s, n=len(p), age_med=p.age.median(), age_q10=p.age.quantile(.1), age_q90=p.age.quantile(.9), female=p.female.mean(), white=(p.RACE == "white").mean(),
                     anchor_med=p.anchor_year.median(), anchor_q25=p.anchor_year.quantile(.25), alive_band=(p.anchor_year > 2015.4).mean(), died=(p.died.mean() if s != "test" else np.nan),
                     events_med=p.n_pre_events.median(), events_q90=p.n_pre_events.quantile(.9), hist_years_med=p.hist_years.median(), first_event_age_med=p.first_event_age.median(), cond_med=p.n_pre_cond.median()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nage at anchor < 1 year:", (pats.age < 1).sum(), "| age < 18:", (pats.age < 18).sum(), "| first recorded event after age 5 (late record start):", (pats.first_event_age > 5).sum())
print("race:", pats.RACE.value_counts(normalize=True).round(3).to_dict())
print("ethnicity:", pats.ETHNICITY.value_counts(normalize=True).round(3).to_dict())

print("\n" + "#" * 100); print("A2. COVARIATE SHIFT: can a classifier tell test from train? val from train? (5-fold AUROC on history features)"); print("#" * 100)
from feats import build_features
d2, X = build_features()
X = X.loc[pats.PATIENT]
def shift_auc(mask_a, mask_b, name):
    m = mask_a | mask_b; y = mask_b[m].astype(int); Xm = X.values[m]
    p = cross_val_predict(lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.05, num_leaves=8, colsample_bytree=0.3, verbose=-1, n_jobs=8), Xm, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    mdl = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.05, num_leaves=8, colsample_bytree=0.3, verbose=-1, n_jobs=8).fit(Xm, y)
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False).head(6)
    print(f"  {name:16s} AUROC={roc_auc_score(y, p):.3f}  top: " + "; ".join(f"{k}={v:.0f}" for k, v in imp.items()))
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
shift_auc(tr, te, "train vs test"); shift_auc(tr, va, "train vs val"); shift_auc(va, te, "val vs test")
# permutation reference: random relabel of train into two halves
rng = np.random.default_rng(0); fake = tr.copy(); idx = np.where(tr)[0]; rng.shuffle(idx); half = np.zeros_like(tr); half[idx[: len(idx) // 2]] = True
shift_auc(tr & ~half, half, "train half A/B (null)")

print("\n" + "#" * 100); print("A3. TABLES: rows, coverage, pre/post anchor, codes (train_val)"); print("#" * 100)
rows = []
for kind in ["COND", "MED", "PROC", "OBS", "ENC", "IMM", "CARE", "ALLERGY", "DEVICE"]:
    e = ev[(ev.kind == kind) & (ev.split != "test")]
    rows.append(dict(table=kind, rows=len(e), patients=e.PATIENT.nunique(), pre_anchor=e.pre.mean(), distinct_codes=e.code.nunique(), codes_ge20pat=(e[e.pre].groupby("code").PATIENT.nunique() >= 20).sum(),
                     rows_per_patient_pre=e[e.pre].groupby("PATIENT").size().median()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
im = pd.concat([pd.read_csv(DATA / f / "imaging_studies.csv", dtype=str) for f in ("train_val", "test")]); im["date"] = _d(im.DATE); im["pre"] = im.date < im.PATIENT.map(pidx.anchor)
print(f"imaging_studies (not used by the project): rows={len(im)} pre-anchor={int(im.pre.sum())} patients(pre)={im[im.pre].PATIENT.nunique()} modalities={im.MODALITY_DESCRIPTION.value_counts().to_dict()}")
enc_cls = ev[(ev.kind == "ENC") & ev.pre].encclass.value_counts(normalize=True).round(3).to_dict(); print("encounter classes (pre-anchor):", enc_cls)

print("\n" + "#" * 100); print("A4. TIME STRUCTURE: wellness cadence, event density by era, record gaps"); print("#" * 100)
enc = ev[(ev.kind == "ENC") & ev.pre & (ev.split != "test")].copy(); enc["year"] = enc.date.dt.year
well = enc[enc.encclass == "wellness"].sort_values("date"); gaps = well.groupby("PATIENT").date.diff().dt.days / 365.25
print("wellness visit gap (years) quantiles:", gaps.quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
dens = ev[ev.pre & (ev.split != "test")].assign(year=lambda x: x.date.dt.year).groupby("year").size()
print("pre-anchor events per calendar decade:", dens.groupby(dens.index // 10 * 10).sum().to_dict())
# per-patient largest gap between consecutive encounters
allenc = enc.sort_values("date"); mg = allenc.groupby("PATIENT").date.diff().dt.days.groupby(allenc.PATIENT).max() / 365.25
print("largest encounter gap per patient (years) quantiles:", mg.quantile([.5, .9, .99]).round(1).to_dict(), "| patients with a gap > 10y:", int((mg > 10).sum()))

print("\n" + "#" * 100); print("A5. CALENDAR-ERA MARKERS: codes whose availability depends on year (potential death/anchor proxies)"); print("#" * 100)
e = ev[ev.pre & (ev.split != "test")].copy(); e["year"] = e.date.dt.year; e["tok"] = e.kind + ":" + e.code
cnt = e.groupby("tok").PATIENT.nunique(); keep = cnt[cnt >= 100].index; e = e[e.tok.isin(keep)]
yr = e.groupby("tok").year.agg(["min", "median", "count"]).join(cnt.rename("patients"))
late = yr[yr["min"] >= 2005].sort_values("patients", ascending=False)
cd = pd.concat([pd.read_csv(DATA / "train_val" / f"{t}.csv", dtype=str, usecols=["CODE", "DESCRIPTION"]).assign(k=k) for t, k in (("conditions", "COND"), ("medications", "MED"), ("procedures", "PROC"), ("observations", "OBS"), ("immunizations", "IMM"), ("careplans", "CARE"))]).drop_duplicates(["k", "CODE"])
cd["tok"] = cd.k + ":" + cd.CODE.str.strip(); dmap = cd.set_index("tok").DESCRIPTION
print("tokens (>=100 patients) that FIRST appear in 2005 or later -> pure era markers:")
for t, r in late.head(25).iterrows(): print(f"   {t:28s} {str(dmap.get(t, ''))[:45]:47s} first_year={int(r['min'])} patients={int(r['patients'])}")
print(f"({len(late)} such tokens of {len(yr)}); tokens that DISAPPEAR before 2000 (last year < 2000):")
last = e.groupby("tok").year.max(); early = yr.join(last.rename("max"))[lambda x: x["max"] < 2000].sort_values("patients", ascending=False)
for t, r in early.head(10).iterrows(): print(f"   {t:28s} {str(dmap.get(t, ''))[:45]:47s} last_year={int(r['max'])} patients={int(r['patients'])}")
# how predictable is the anchor year from token presence alone (no QoL)?
tokp = e[~e.code.isin(["QALY", "DALY", "QOLS"])][["PATIENT", "tok"]].drop_duplicates(); M = pd.crosstab(tokp.PATIENT, tokp.tok).reindex(pats[pats.split != "test"].PATIENT).fillna(0)
yv = pidx.anchor_year.reindex(M.index).values
pr = cross_val_predict(lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3, verbose=-1, n_jobs=8), M.values, yv, cv=5)
print(f"anchor year from token PRESENCE only (no QoL, no counts): CV R2={1-((pr-yv)**2).mean()/yv.var():.3f}  MAE={np.abs(pr-yv).mean():.1f} years")
dv = pidx.died.reindex(M.index).values.astype(int)
pd_ = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3, verbose=-1, n_jobs=8), M.values, dv, cv=5, method="predict_proba")[:, 1]
print(f"death-in-window from token PRESENCE only: CV AUROC={roc_auc_score(dv, pd_):.3f}")

print("\n" + "#" * 100); print("A6. NUMERIC OBSERVATIONS: units, ranges, coverage"); print("#" * 100)
num = ev[(ev.kind == "OBS") & ev.num.notna() & ev.pre & (ev.split != "test")]
od = cd[cd.k == "OBS"].set_index("CODE").DESCRIPTION
st = num.groupby("code").agg(patients=("PATIENT", "nunique"), rows=("num", "size"), units=("units", lambda s: ",".join(sorted(set(s))[:3])), p01=("num", lambda s: s.quantile(.01)), med=("num", "median"), p99=("num", lambda s: s.quantile(.99))).sort_values("patients", ascending=False)
st.insert(0, "desc", [str(od.get(c, ""))[:30] for c in st.index]); print(st.head(30).round(2).to_string())
print("codes with >1 unit:", int((num.groupby("code").units.nunique() > 1).sum()))
cat = ev[(ev.kind == "OBS") & ev.num.isna() & ev.pre & (ev.split != "test")]
print("categorical observation codes:", cat.code.nunique(), "| top:", cat.groupby("code").PATIENT.nunique().sort_values(ascending=False).head(6).to_dict())
