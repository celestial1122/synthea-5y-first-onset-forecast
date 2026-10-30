"""Person-period (discrete-time hazard) training for the 30 non-terminal targets.
Every 2 years before the official anchor is a landmark; features are computed from the history before that landmark and the
label is 'first diagnosis of k in (landmark, landmark+5y]' (fully observed for train patients). Grouped 5-fold by patient:
train on all landmarks of training patients, predict the OFFICIAL anchor of held-out patients -> OOF comparable with the
official-only model. Terminal targets are excluded (their official windows are death-aligned, see diagnosis doc §3.3)."""
import time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "pp2"; OUT.mkdir(exist_ok=True)
d = load(); pats, events, codes, targets = d["pats"], d["events"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
Ft = d["first_targets"]; pidx = pats.set_index("PATIENT")
train_ids = pats[pats.split == "train"].PATIENT.values; val_ids = pats[pats.split == "val"].PATIENT.values; test_ids = pats[pats.split == "test"].PATIENT.values
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
NONTERM = [c for c in codes if c not in TERMINAL]
ev = events[["PATIENT", "date", "kind", "code", "value", "units", "num", "encclass", "split"]].copy()
is_obs_cat = (ev.kind == "OBS") & ev.num.isna(); ev["tok"] = ev.kind + ":" + ev.code
ev.loc[is_obs_cat, "tok"] = "OBS:" + ev.loc[is_obs_cat, "code"] + "|" + ev.loc[is_obs_cat, "value"].fillna("").astype(str)
ev.loc[ev.kind == "ENC", "tok"] = "ENC:" + ev.loc[ev.kind == "ENC", "encclass"].astype(str)
ev["is_num"] = (ev.kind == "OBS") & ev.num.notna(); ev.loc[ev.is_num, "code"] = ev.loc[ev.is_num, "code"] + "|" + ev.loc[ev.is_num, "units"].fillna("").astype(str)
pre0 = ev[ev.date < ev.PATIENT.map(pidx.anchor)]
sup = pre0[~pre0.is_num & (pre0.split == "train")][["PATIENT", "tok"]].drop_duplicates().tok.value_counts(); KEEP = sup[sup >= 20].index
nsup = pre0[pre0.is_num & (pre0.split == "train")].groupby("code").PATIENT.nunique(); NKEEP = nsup[nsup >= 50].index

def features(ids, landmark):
    lm = landmark.reindex(ids); e = ev[ev.PATIENT.isin(ids)].copy(); e["lm"] = e.PATIENT.map(lm); e = e[e.date < e.lm]; e["yb"] = (e.lm - e.date).dt.days / 365.25
    sub = e[e.tok.isin(KEEP) & ~e.is_num]; g = sub.groupby(["PATIENT", "tok"])
    fr = [g.size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("cnt_"),
          sub[sub.yb <= 5].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("c5_"),
          g.yb.min().unstack().reindex(index=ids, columns=KEEP).add_prefix("rec_"), g.yb.max().unstack().reindex(index=ids, columns=KEEP).add_prefix("first_")]
    num = e[e.is_num & e.code.isin(NKEEP)].sort_values("date"); gn = num.groupby(["PATIENT", "code"]).num
    fr += [gn.last().unstack().reindex(index=ids, columns=NKEEP).add_prefix("last_"), gn.max().unstack().reindex(index=ids, columns=NKEEP).add_prefix("max_"), gn.min().unstack().reindex(index=ids, columns=NKEEP).add_prefix("min_")]
    st = pd.DataFrame(index=ids); st["age"] = ((lm - pidx.BIRTHDATE.reindex(ids)).dt.days / 365.25).values; st["female"] = pidx.female.reindex(ids).values
    for r in ("white", "black", "asian", "hispanic"): st[f"race_{r}"] = (pidx.RACE.reindex(ids) == r).astype(int).values
    gp = e.groupby("PATIENT"); st["hist_years"] = gp.yb.max().reindex(ids).fillna(0); st["n_events"] = gp.size().reindex(ids).fillna(0); st["n_events_5y"] = e[e.yb <= 5].groupby("PATIENT").size().reindex(ids).fillna(0)
    st["n_cond_codes"] = e[e.kind == "COND"].groupby("PATIENT").code.nunique().reindex(ids).fillna(0)
    Fdt = Ft.reindex(ids); isna = Fdt.isna().values; F = Fdt.values.astype("datetime64[D]").astype(float); F[isna] = np.nan
    L = lm.values.astype("datetime64[D]").astype(float)[:, None]
    prior = (F <= L) & ~np.isnan(F); y = (F > L) & (F <= L + 1826.25) & ~np.isnan(F)
    X = pd.concat([st, pd.DataFrame(prior.astype(np.float32), index=ids, columns=codes).add_prefix("prior_")] + fr, axis=1).astype(np.float32)
    return X, y.astype(np.int8), prior

# official rows for all splits (k=0) and landmark rows for train patients (k = 2, 4, ..., 50 years before anchor)
X0, y0, p0 = features(np.concatenate([train_ids, val_ids, test_ids]), pidx.anchor)
Yoff = d["Y"].reindex(X0.index).values; Poff = d["P"].reindex(X0.index).values.astype(bool)
rows = [(X0.loc[train_ids], Yoff[: len(train_ids)], Poff[: len(train_ids)], np.zeros(len(train_ids)))]
for k in range(1, 51, 1):
    lm = pidx.anchor - pd.DateOffset(years=k)
    ok = train_ids[((lm.reindex(train_ids) - pidx.BIRTHDATE.reindex(train_ids)).dt.days > 365).values]
    if len(ok) < 50: break
    Xk, yk, pk = features(ok, lm); has_hist = Xk.n_events.values > 0
    rows.append((Xk[has_hist], yk[has_hist], pk[has_hist], np.full(int(has_hist.sum()), k)))
    print(f"  landmark -{k}y: {int(has_hist.sum())} rows, positives {int(yk[has_hist].sum())}  [{time.time()-t0:.0f}s]", flush=True)
Xpp = pd.concat([r[0] for r in rows]); ypp = np.concatenate([r[1] for r in rows]); ppp = np.concatenate([r[2] for r in rows]); kpp = np.concatenate([r[3] for r in rows])
pid_pp = Xpp.index.values; print("person-period rows:", len(Xpp), "features:", Xpp.shape[1], f"[{time.time()-t0:.0f}s]", flush=True)
import xgboost as xgb
CFG = dict(device="cuda", tree_method="hist", n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.3, reg_lambda=5.0, min_child_weight=5, verbosity=0)
Xtr0 = X0.loc[train_ids].values; Xva0 = X0.loc[val_ids].values; Xte0 = X0.loc[test_ids].values; Ytr = Yoff[: len(train_ids)]; Ptr = Poff[: len(train_ids)]; Yva = Yoff[len(train_ids): len(train_ids) + len(val_ids)]; Pva = Poff[len(train_ids): len(train_ids) + len(val_ids)]
def aps(p, Yv, Pv, idx):
    p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in idx])
nti = [codes.index(c) for c in NONTERM]
res = {}
for name, weight_off in (("pp yearly uniform", 1.0), ("pp yearly decay", "decay")):
    oof = np.zeros((len(train_ids), 40)); pv = np.zeros((len(val_ids), 40)); pt = np.zeros((len(test_ids), 40))
    for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train_ids)):
        tr_set = set(train_ids[a])
        if weight_off is None:
            Xf, yf, pf, wf = Xtr0[a], Ytr[a], Ptr[a], None
        else:
            m = np.isin(pid_pp, list(tr_set)); Xf, yf, pf = Xpp.values[m], ypp[m], ppp[m]; wf = (0.5 ** (kpp[m] / 10.0)) if weight_off == "decay" else np.where(kpp[m] == 0, weight_off, 1.0)
        for j in nti:
            at = ~pf[:, j]; y = yf[at, j]
            if y.sum() < 5: continue
            mdl = xgb.XGBClassifier(**CFG).fit(Xf[at], y, sample_weight=None if wf is None else wf[at])
            oof[b, j] = mdl.predict_proba(Xtr0[b])[:, 1]; pv[:, j] += mdl.predict_proba(Xva0)[:, 1] / 5; pt[:, j] += mdl.predict_proba(Xte0)[:, 1] / 5
        print(f"    {name}: fold {fold} done [{time.time()-t0:.0f}s]", flush=True)
    a_oof = aps(oof, Ytr, Ptr, nti); a_val = aps(pv, Yva, Pva, nti); res[name] = (oof, pv, pt, a_oof, a_val)
    print(f"{name:28s} non-terminal 30 targets: OOF mean AP={a_oof.mean():.4f} | val {a_val.mean():.4f}", flush=True)
per = pd.DataFrame({"cond": [desc[c][:32] for c in NONTERM], "train_pos": [int(Ytr[~Ptr[:, j], j].sum()) for j in nti], **{k: v[3] for k, v in res.items()}}, index=NONTERM).round(3)
pd.set_option("display.width", 220); print(per.to_string()); per.to_csv(OUT / "per_condition_pp.csv")
for k, v in res.items():
    tag = k.replace(" ", "_"); np.save(OUT / f"{tag}_oof.npy", v[0]); np.save(OUT / f"{tag}_val.npy", v[1]); np.save(OUT / f"{tag}_test.npy", v[2])
json.dump({k: {"oof": float(v[3].mean()), "val": float(v[4].mean())} for k, v in res.items()}, open(OUT / "summary_pp.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
