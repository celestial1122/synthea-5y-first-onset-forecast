"""Tabular analogue of dense supervision: multiple landmarks per training patient (anchor - k years),
each with its own 5-year first-diagnosis labels. Evaluate on the official val anchors only.
Also: how predictable is 'death within the window' from pre-anchor history alone?
"""
import time, warnings, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from ehr_load import load
warnings.filterwarnings("ignore")
t0 = time.time()
d = load()
pats, events, codes, targets = d["pats"], d["events"], d["codes"], d["targets"]
desc = dict(zip(targets.CODE, targets.DESCRIPTION)); short = {c: desc[c][:34] for c in codes}
Ft = d["first_targets"]
train_ids = pats[pats.split == "train"].PATIENT.values; val_ids = pats[pats.split == "val"].PATIENT.values
pidx = pats.set_index("PATIENT")
ev = events[["PATIENT", "date", "kind", "code", "value", "num", "encclass", "split"]].copy()
is_obs_cat = (ev.kind == "OBS") & ev.num.isna()
ev["tok"] = ev.kind + ":" + ev.code
ev.loc[is_obs_cat, "tok"] = "OBS:" + ev.loc[is_obs_cat, "code"] + "|" + ev.loc[is_obs_cat, "value"].fillna("").astype(str)
ev.loc[ev.kind == "ENC", "tok"] = "ENC:" + ev.loc[ev.kind == "ENC", "encclass"].astype(str)
ev["is_num"] = (ev.kind == "OBS") & ev.num.notna()

# fixed feature vocabulary from official train rows
pre0 = ev[ev.date < ev.PATIENT.map(pidx.anchor)]
sup = pre0[~pre0.is_num & (pre0.split == "train")][["PATIENT", "tok"]].drop_duplicates().tok.value_counts()
KEEP = sup[sup >= 20].index
nsup = pre0[pre0.is_num & (pre0.split == "train")].groupby("code").PATIENT.nunique()
NKEEP = nsup[nsup >= 50].index
print("vocab", len(KEEP), "numeric", len(NKEEP))


def features(ids, landmark):
    """ids: patients; landmark: Series PATIENT->date. Features from events strictly before landmark."""
    lm = landmark.reindex(ids)
    e = ev[ev.PATIENT.isin(ids)].copy()
    e["lm"] = e.PATIENT.map(lm)
    e = e[e.date < e.lm]
    e["yb"] = (e.lm - e.date).dt.days / 365.25
    sub = e[e.tok.isin(KEEP) & ~e.is_num]
    g = sub.groupby(["PATIENT", "tok"])
    parts = {
        "cnt_": g.size().unstack(fill_value=0),
        "c5_": sub[sub.yb <= 5].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0),
        "c1_": sub[sub.yb <= 1].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0),
        "rec_": g.yb.min().unstack(),
    }
    frames = [v.reindex(index=ids, columns=KEEP).fillna(0 if k != "rec_" else np.nan).add_prefix(k) for k, v in parts.items()]
    num = e[e.is_num & e.code.isin(NKEEP)].sort_values("date")
    gn = num.groupby(["PATIENT", "code"]).num
    frames += [gn.last().unstack().reindex(index=ids, columns=NKEEP).add_prefix("last_"),
               gn.max().unstack().reindex(index=ids, columns=NKEEP).add_prefix("max_"),
               gn.min().unstack().reindex(index=ids, columns=NKEEP).add_prefix("min_"),
               num[num.yb <= 5].groupby(["PATIENT", "code"]).num.mean().unstack().reindex(index=ids, columns=NKEEP).add_prefix("m5_")]
    st = pd.DataFrame(index=ids)
    st["age"] = ((lm - pidx.BIRTHDATE.reindex(ids)).dt.days / 365.25).values
    st["female"] = pidx.female.reindex(ids).values
    gp = e.groupby("PATIENT")
    st["hist_years"] = gp.yb.max().reindex(ids).fillna(0)
    st["n_events"] = gp.size().reindex(ids).fillna(0)
    st["n_events_5y"] = e[e.yb <= 5].groupby("PATIENT").size().reindex(ids).fillna(0)
    st["n_cond_codes"] = e[e.kind == "COND"].groupby("PATIENT").code.nunique().reindex(ids).fillna(0)
    st["n_med_codes"] = e[e.kind == "MED"].groupby("PATIENT").code.nunique().reindex(ids).fillna(0)
    for r in ("white", "black", "asian", "hispanic"):
        st[f"race_{r}"] = (pidx.RACE.reindex(ids) == r).astype(int).values
    # labels / prior at this landmark: window (lm, lm + 5y]
    Fdt = Ft.reindex(ids); isna = Fdt.isna().values
    F = Fdt.values.astype("datetime64[D]").astype(float); F[isna] = np.nan   # NaT -> NaN (never diagnosed)
    L = lm.values.astype("datetime64[D]").astype(float)[:, None]
    prior = (F <= L) & ~np.isnan(F)
    y = (F > L) & (F <= L + 1826.25) & ~np.isnan(F)
    assert prior.mean() < 0.2 and y.mean() < 0.1
    pri = pd.DataFrame(prior.astype(int), index=ids, columns=codes).add_prefix("prior_")
    X = pd.concat([st, pri] + frames, axis=1).astype(np.float32)
    return X, y.astype(int), prior


# official anchors
Xtr0, ytr0, ptr0 = features(train_ids, pidx.anchor)
Xva, yva, pva = features(val_ids, pidx.anchor)
# the official definition uses [anchor, end] with calendar years; check agreement with (lm, lm+5y] version
Yoff = d["Y"].loc[train_ids].values; print("official-vs-landmark label agreement on train:", (Yoff == ytr0).mean().round(5))
ytr0 = Yoff; ptr0 = d["P"].loc[train_ids].values.astype(bool)
yva = d["Y"].loc[val_ids].values; pva = d["P"].loc[val_ids].values.astype(bool)

LGB = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.3, reg_lambda=5.0, n_estimators=350, verbose=-1, random_state=0, n_jobs=8)

def fit_eval(Xs, ys, ps, ws, name, extra_val_cols=None):
    cols = Xs.columns
    Xv = Xva.reindex(columns=cols)
    if extra_val_cols:
        for k, v in extra_val_cols.items(): Xv[k] = v
    probs = np.zeros((len(Xv), len(codes)))
    for j, c in enumerate(codes):
        m = ~ps[:, j]; y = ys[m, j]
        if y.sum() < 3:
            probs[:, j] = y.mean(); continue
        mdl = lgb.LGBMClassifier(**LGB).fit(Xs.values[m], y, sample_weight=None if ws is None else ws[m])
        probs[:, j] = mdl.predict_proba(Xv.values)[:, 1]
    probs[pva] = 0
    aps = [average_precision_score(yva[:, j], probs[:, j]) if yva[:, j].sum() else 0 for j in range(len(codes))]
    aucs = [roc_auc_score(yva[:, j], probs[:, j]) for j in range(len(codes)) if 0 < yva[:, j].sum() < len(yva)]
    print(f"{name:70s} val mAP={np.mean(aps):.4f} macroAUROC={np.mean(aucs):.4f} (rows={len(Xs)}) [{time.time()-t0:.0f}s]", flush=True)
    return np.array(aps), probs

ap_off, p_off = fit_eval(Xtr0, ytr0, ptr0, None, "official anchors only (baseline)")

# augmented landmarks k = 1..K years before the official anchor
K = 10
Xs, ys, ps, ks = [Xtr0], [ytr0], [ptr0], [np.zeros(len(Xtr0))]
for k in range(1, K + 1):
    lm = pidx.anchor - pd.DateOffset(years=k)
    birth = pidx.BIRTHDATE
    ok = train_ids[((lm.reindex(train_ids) - birth.reindex(train_ids)).dt.days > 365).values]
    Xk, yk, pk = features(ok, lm)
    Xs.append(Xk); ys.append(yk); ps.append(pk); ks.append(np.full(len(Xk), k))
    print(f"  landmark -{k}y: {len(ok)} patients, positives={yk.sum()} [{time.time()-t0:.0f}s]", flush=True)
Xall = pd.concat(Xs); yall = np.concatenate(ys); pall = np.concatenate(ps); kall = np.concatenate(ks)
for K_use in (3, 10):
    sel = kall <= K_use
    fit_eval(Xall[sel], yall[sel], pall[sel], None, f"official + landmarks up to -{K_use}y, uniform weight")
sel = kall <= 10
w = np.where(kall == 0, 5.0, 1.0)
fit_eval(Xall[sel], yall[sel], pall[sel], w, "official + landmarks -10y, official rows weight 5")
Xo = Xall.copy(); Xo["official"] = (kall == 0).astype(np.float32)
fit_eval(Xo, yall, pall, None, "official + landmarks -10y, with 'official' indicator feature (val=1)", extra_val_cols={"official": 1.0})

# how predictable is death-in-window from pre-anchor history?
died_tr = pidx.died.reindex(train_ids).values.astype(int); died_va = pidx.died.reindex(val_ids).values.astype(int)
mdl = lgb.LGBMClassifier(**LGB).fit(Xtr0.values, died_tr)
pd_va = mdl.predict_proba(Xva.values)[:, 1]
print(f"\nDeath-in-window from pre-anchor history only: val AUROC={roc_auc_score(died_va, pd_va):.3f}  AP={average_precision_score(died_va, pd_va):.3f} (prevalence {died_va.mean():.3f})")
imp = pd.Series(mdl.booster_.feature_importance("gain"), index=Xtr0.columns).sort_values(ascending=False).head(10)
print("top death predictors:", "; ".join(f"{k}={v:.0f}" for k, v in imp.items()))
# and with the calendar anchor year
Xtr_a = Xtr0.copy(); Xtr_a["anchor_year"] = pidx.anchor_year.reindex(train_ids).values
Xva_a = Xva.copy(); Xva_a["anchor_year"] = pidx.anchor_year.reindex(val_ids).values
mdl = lgb.LGBMClassifier(**LGB).fit(Xtr_a.values, died_tr)
print(f"Death-in-window with anchor_year added:          val AUROC={roc_auc_score(died_va, mdl.predict_proba(Xva_a.values)[:,1]):.3f}")
print("done", f"{time.time()-t0:.0f}s")
