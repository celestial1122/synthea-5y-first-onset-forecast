"""Strong tabular baseline (hand-crafted history features + per-target LightGBM / logistic regression).
Trains on official train only, evaluates on official val with the project's prior->0 masking.
Also 5-fold out-of-fold CV on train for a lower-variance estimate.
"""
import json, time, warnings, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ehr_load import load
warnings.filterwarnings("ignore")
t0 = time.time()
d = load()
pats, events, Y, P, codes, targets = d["pats"], d["events"], d["Y"], d["P"], d["codes"], d["targets"]
desc = dict(zip(targets.CODE, targets.DESCRIPTION)); short = {c: desc[c][:34] for c in codes}
train = pats[pats.split == "train"]; val = pats[pats.split == "val"]; test = pats[pats.split == "test"]
train_ids = train.PATIENT.values

# ---------------- features (pre-anchor only) ----------------
pre = events[events.pre].copy()
is_obs_cat = (pre.kind == "OBS") & pre.num.isna()
pre["tok"] = pre.kind + ":" + pre.code
pre.loc[is_obs_cat, "tok"] = "OBS:" + pre.loc[is_obs_cat, "code"] + "|" + pre.loc[is_obs_cat, "value"].fillna("").astype(str)
pre.loc[pre.kind == "ENC", "tok"] = "ENC:" + pre.loc[pre.kind == "ENC", "encclass"].astype(str)
is_obs_num = (pre.kind == "OBS") & pre.num.notna()
tok_pat = pre.loc[~is_obs_num, ["PATIENT", "tok"]].drop_duplicates()
support = tok_pat[tok_pat.PATIENT.isin(train_ids)].tok.value_counts()
keep = support[support >= 20].index
print(f"token features kept: {len(keep)} (support>=20 train patients)")
sub = pre[pre.tok.isin(keep) & ~is_obs_num]
g = sub.groupby(["PATIENT", "tok"])
cnt_all = g.size().unstack(fill_value=0)
cnt_5y = sub[sub.years_before <= 5].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0)
cnt_1y = sub[sub.years_before <= 1].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0)
recency = g.years_before.min().unstack()
idx = pats.PATIENT
cnt_all = cnt_all.reindex(index=idx, columns=keep, fill_value=0)
cnt_5y = cnt_5y.reindex(index=idx, columns=keep, fill_value=0)
cnt_1y = cnt_1y.reindex(index=idx, columns=keep, fill_value=0)
recency = recency.reindex(index=idx, columns=keep)  # NaN = never
num = pre[is_obs_num].sort_values("date")
nsupport = num[num.PATIENT.isin(train_ids)].groupby("code").PATIENT.nunique()
nkeep = nsupport[nsupport >= 50].index
num = num[num.code.isin(nkeep)]
gn = num.groupby(["PATIENT", "code"]).num
num_last = gn.last().unstack().reindex(index=idx, columns=nkeep)
num_max = gn.max().unstack().reindex(index=idx, columns=nkeep)
num_min = gn.min().unstack().reindex(index=idx, columns=nkeep)
num_last5 = num[num.years_before <= 5].groupby(["PATIENT", "code"]).num.mean().unstack().reindex(index=idx, columns=nkeep)
print(f"numeric observation codes kept: {len(nkeep)}")
gp = pre.groupby("PATIENT")
static = pd.DataFrame(index=idx)
static["age"] = pats.set_index("PATIENT").age
static["female"] = pats.set_index("PATIENT").female
static["hist_years"] = gp.years_before.max().reindex(idx).fillna(0)
static["n_events"] = gp.size().reindex(idx).fillna(0)
static["n_events_5y"] = pre[pre.years_before <= 5].groupby("PATIENT").size().reindex(idx).fillna(0)
static["n_cond_codes"] = pre[pre.kind == "COND"].groupby("PATIENT").code.nunique().reindex(idx).fillna(0)
static["n_med_codes"] = pre[pre.kind == "MED"].groupby("PATIENT").code.nunique().reindex(idx).fillna(0)
static["anchor_year"] = pats.set_index("PATIENT").anchor_year
for r in ("white", "black", "asian", "hispanic"):
    static[f"race_{r}"] = (pats.set_index("PATIENT").RACE == r).astype(int)
prior_df = P.add_prefix("prior_")

def assemble(use_anchor_year):
    parts = [static.drop(columns=[] if use_anchor_year else ["anchor_year"]), prior_df,
             cnt_all.add_prefix("cnt_"), cnt_5y.add_prefix("c5_"), cnt_1y.add_prefix("c1_"), recency.add_prefix("rec_"),
             num_last.add_prefix("last_"), num_max.add_prefix("max_"), num_min.add_prefix("min_"), num_last5.add_prefix("m5_")]
    X = pd.concat(parts, axis=1).astype(np.float32)
    return X

X_full = assemble(True)
print("feature matrix:", X_full.shape, f"({time.time()-t0:.0f}s)")
feature_names = X_full.columns.tolist()
Xtr, Xva, Xte = X_full.loc[train.PATIENT], X_full.loc[val.PATIENT], X_full.loc[test.PATIENT]
Ytr, Ptr = Y.loc[train.PATIENT], P.loc[train.PATIENT]
Yva, Pva = Y.loc[val.PATIENT], P.loc[val.PATIENT]

LGB = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.3, reg_lambda=5.0, n_estimators=350, verbose=-1, random_state=0, n_jobs=8)

def fit_lgb(X, y, params=LGB):
    m = lgb.LGBMClassifier(**params); m.fit(X, y); return m

def fit_lr(X, y, C=0.02):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y)
    return sc, m

def evaluate(name, probs, Yv=Yva, Pv=Pva, verbose=True):
    probs = np.where(Pv.values == 1, 0.0, probs)
    aps, aucs = [], []
    for j, c in enumerate(codes):
        yv = Yv[c].values
        aps.append(average_precision_score(yv, probs[:, j]) if yv.sum() else 0.0)
        aucs.append(roc_auc_score(yv, probs[:, j]) if 0 < yv.sum() < len(yv) else np.nan)
    if verbose:
        print(f"{name:60s} mAP={np.mean(aps):.4f} macroAUROC={np.nanmean(aucs):.4f}")
    return np.array(aps), np.array(aucs)

def run_models(Xtr, Xva, drop=()):
    cols = [c for c in Xtr.columns if c not in drop]
    pg = np.zeros((len(Xva), len(codes))); pl = np.zeros_like(pg)
    Xl_tr = Xtr[cols].fillna(0).values; Xl_va = Xva[cols].fillna(0).values
    for j, c in enumerate(codes):
        m = Ptr[c].values == 0; y = Ytr[c].values[m]
        if y.sum() < 3:
            pg[:, j] = pl[:, j] = y.mean(); continue
        pg[:, j] = fit_lgb(Xtr[cols].values[m], y).predict_proba(Xva[cols].values)[:, 1]
        sc, lr = fit_lr(Xl_tr[m], y)
        pl[:, j] = lr.predict_proba(sc.transform(Xl_va))[:, 1]
    return pg, pl

results = {}
print("\n=== OFFICIAL VAL (train on official train) ===")
pg, pl = run_models(Xtr, Xva)
ap_g, auc_g = evaluate("LightGBM (all features incl. anchor_year)", pg)
ap_l, auc_l = evaluate("LogReg L2 (all features incl. anchor_year)", pl)
ap_e, auc_e = evaluate("avg(LightGBM, LogReg)", (pg + pl) / 2)
results["val_full"] = dict(lgb=float(ap_g.mean()), lr=float(ap_l.mean()), ens=float(ap_e.mean()), lgb_auc=float(np.nanmean(auc_g)))
pg0, pl0 = run_models(Xtr, Xva, drop=("anchor_year",))
ap_g0, auc_g0 = evaluate("LightGBM (NO anchor_year)", pg0)
ap_l0, auc_l0 = evaluate("LogReg L2 (NO anchor_year)", pl0)
ap_e0, auc_e0 = evaluate("avg(LightGBM, LogReg) (NO anchor_year)", (pg0 + pl0) / 2)
results["val_no_anchor"] = dict(lgb=float(ap_g0.mean()), lr=float(ap_l0.mean()), ens=float(ap_e0.mean()), lgb_auc=float(np.nanmean(auc_g0)))
# static-only GBM for reference (age/sex/history size, no codes)
static_cols = [c for c in static.columns if c != "anchor_year"]
pgs, pls = run_models(Xtr[static_cols], Xva[static_cols])
ap_s, _ = evaluate("LightGBM static only (age, sex, history size, race)", pgs)

# transformer per-condition reference (strict C0R0 three seeds, mean AP)
import glob
tf = []
for p in glob.glob(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\artifacts\experimental_audit_strict\runs\C0R0-seed*-repeat-a\per_condition.csv"):
    tf.append(pd.read_csv(p, dtype={"code": str}).set_index("code").average_precision)
tf_ap = pd.concat(tf, axis=1).mean(axis=1).reindex(codes) if tf else pd.Series(np.nan, index=codes)
cmp = pd.DataFrame({"cond": [short[c] for c in codes], "val_pos": Yva.sum().values, "transformer_C0R0": tf_ap.values,
                    "lgb_static": ap_s, "lgb_noanchor": ap_g0, "lr_noanchor": ap_l0, "ens_noanchor": ap_e0, "lgb_anchor": ap_g, "ens_anchor": ap_e}, index=codes).round(3)
pd.set_option("display.width", 250)
print("\nper-condition AP on val:")
print(cmp.to_string())
print("\ncolumn means:", cmp.drop(columns=["cond", "val_pos"]).mean().round(4).to_dict())
cmp.to_csv("tabular_per_condition_val.csv")

# feature importance summary for a few targets
print("\nTop LightGBM features (gain) for selected targets, no-anchor model:")
cols0 = [c for c in Xtr.columns if c != "anchor_year"]
for c in ["162864005", "15777000", "443165006", "92691004", "254637007", "26929004", "22298006", "72892002", "233604007"]:
    m = Ptr[c].values == 0; y = Ytr[c].values[m]
    mdl = fit_lgb(Xtr[cols0].values[m], y)
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=cols0).sort_values(ascending=False).head(6)
    print(f"  {short[c]:36s} " + "; ".join(f"{k}={v:.0f}" for k, v in imp.items()))

print(f"\n=== 5-fold OOF CV on official TRAIN (2791 patients), LightGBM no-anchor vs with-anchor  ({time.time()-t0:.0f}s) ===")
kf = KFold(5, shuffle=True, random_state=0)
oof0 = np.zeros((len(Xtr), len(codes))); oof1 = np.zeros_like(oof0); oofl = np.zeros_like(oof0)
for fold, (a, b) in enumerate(kf.split(Xtr)):
    for j, c in enumerate(codes):
        m = Ptr[c].values[a] == 0; y = Ytr[c].values[a][m]
        if y.sum() < 3:
            oof0[b, j] = oof1[b, j] = oofl[b, j] = y.mean(); continue
        oof0[b, j] = fit_lgb(Xtr[cols0].values[a][m], y).predict_proba(Xtr[cols0].values[b])[:, 1]
        oof1[b, j] = fit_lgb(Xtr.values[a][m], y).predict_proba(Xtr.values[b])[:, 1]
        sc, lr = fit_lr(Xtr[cols0].fillna(0).values[a][m], y)
        oofl[b, j] = lr.predict_proba(sc.transform(Xtr[cols0].fillna(0).values[b]))[:, 1]
    print(f"  fold {fold} done ({time.time()-t0:.0f}s)")
a0, _ = evaluate("OOF LightGBM (NO anchor_year)", oof0, Ytr, Ptr)
al, _ = evaluate("OOF LogReg (NO anchor_year)", oofl, Ytr, Ptr)
ae, _ = evaluate("OOF avg(LightGBM, LogReg) (NO anchor_year)", (oof0 + oofl) / 2, Ytr, Ptr)
a1, _ = evaluate("OOF LightGBM (with anchor_year)", oof1, Ytr, Ptr)
results["oof_train"] = dict(lgb_noanchor=float(a0.mean()), lr_noanchor=float(al.mean()), ens_noanchor=float(ae.mean()), lgb_anchor=float(a1.mean()))
# bootstrap SE of mAP on val vs OOF to show the noise floor
rng = np.random.default_rng(0)
def boot_map(probs, Yv, Pv, n=300):
    probs = np.where(Pv.values == 1, 0.0, probs); vals = []
    for _ in range(n):
        ii = rng.integers(0, len(probs), len(probs))
        aps = [average_precision_score(Yv[c].values[ii], probs[ii, j]) if Yv[c].values[ii].sum() else 0.0 for j, c in enumerate(codes)]
        vals.append(np.mean(aps))
    return float(np.std(vals))
print(f"\nbootstrap SD of mAP: official val (365) = {boot_map((pg0+pl0)/2, Yva, Pva):.4f} ; train OOF (2791) = {boot_map((oof0+oofl)/2, Ytr, Ptr):.4f}")
json.dump(results, open("tabular_results.json", "w"), indent=1)
np.save("tabular_val_probs_noanchor.npy", (pg0 + pl0) / 2)
np.save("tabular_val_probs_anchor.npy", (pg + pl) / 2)
print("done", f"{time.time()-t0:.0f}s")
