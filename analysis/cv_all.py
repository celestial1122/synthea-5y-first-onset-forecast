"""Is the official val easier than train? 5-fold CV over all 3156 train+val patients with the same GBM;
report OOF mAP on the val subset vs the train subset (same models, different evaluation subsets),
plus per-condition OOF AP (train-only CV) vs official-val AP to locate the gap."""
import warnings, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore")
import importlib.util, sys
spec = importlib.util.spec_from_file_location("aug", str(__import__("pathlib").Path(__file__).parent / "augment_landmarks.py"))
# reuse feature builder without running the experiment: copy the relevant code instead of importing
t0 = time.time()
d = load(); pats, events, codes, targets = d["pats"], d["events"], d["codes"], d["targets"]
Ft = d["first_targets"]; pidx = pats.set_index("PATIENT")
ids_all = pats[pats.split != "test"].PATIENT.values
is_val = (pats[pats.split != "test"].split == "val").values
ev = events[["PATIENT", "date", "kind", "code", "value", "num", "encclass", "split"]].copy()
is_obs_cat = (ev.kind == "OBS") & ev.num.isna()
ev["tok"] = ev.kind + ":" + ev.code
ev.loc[is_obs_cat, "tok"] = "OBS:" + ev.loc[is_obs_cat, "code"] + "|" + ev.loc[is_obs_cat, "value"].fillna("").astype(str)
ev.loc[ev.kind == "ENC", "tok"] = "ENC:" + ev.loc[ev.kind == "ENC", "encclass"].astype(str)
ev["is_num"] = (ev.kind == "OBS") & ev.num.notna()
pre = ev[ev.date < ev.PATIENT.map(pidx.anchor)].copy()
pre["yb"] = (pre.PATIENT.map(pidx.anchor) - pre.date).dt.days / 365.25
sup = pre[~pre.is_num & (pre.split == "train")][["PATIENT", "tok"]].drop_duplicates().tok.value_counts(); KEEP = sup[sup >= 20].index
nsup = pre[pre.is_num & (pre.split == "train")].groupby("code").PATIENT.nunique(); NKEEP = nsup[nsup >= 50].index
sub = pre[pre.tok.isin(KEEP) & ~pre.is_num]; g = sub.groupby(["PATIENT", "tok"])
frames = [g.size().unstack(fill_value=0).reindex(index=ids_all, columns=KEEP).fillna(0).add_prefix("cnt_"),
          sub[sub.yb <= 5].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids_all, columns=KEEP).fillna(0).add_prefix("c5_"),
          sub[sub.yb <= 1].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids_all, columns=KEEP).fillna(0).add_prefix("c1_"),
          g.yb.min().unstack().reindex(index=ids_all, columns=KEEP).add_prefix("rec_")]
num = pre[pre.is_num & pre.code.isin(NKEEP)].sort_values("date"); gn = num.groupby(["PATIENT", "code"]).num
frames += [gn.last().unstack().reindex(index=ids_all, columns=NKEEP).add_prefix("last_"), gn.max().unstack().reindex(index=ids_all, columns=NKEEP).add_prefix("max_"),
           gn.min().unstack().reindex(index=ids_all, columns=NKEEP).add_prefix("min_"),
           num[num.yb <= 5].groupby(["PATIENT", "code"]).num.mean().unstack().reindex(index=ids_all, columns=NKEEP).add_prefix("m5_")]
st = pd.DataFrame(index=ids_all); st["age"] = pidx.age.reindex(ids_all); st["female"] = pidx.female.reindex(ids_all)
gp = pre.groupby("PATIENT"); st["hist_years"] = gp.yb.max().reindex(ids_all).fillna(0); st["n_events"] = gp.size().reindex(ids_all).fillna(0)
st["n_events_5y"] = pre[pre.yb <= 5].groupby("PATIENT").size().reindex(ids_all).fillna(0)
st["n_cond_codes"] = pre[pre.kind == "COND"].groupby("PATIENT").code.nunique().reindex(ids_all).fillna(0)
st["n_med_codes"] = pre[pre.kind == "MED"].groupby("PATIENT").code.nunique().reindex(ids_all).fillna(0)
for r in ("white", "black", "asian", "hispanic"): st[f"race_{r}"] = (pidx.RACE.reindex(ids_all) == r).astype(int)
X = pd.concat([st, d["P"].loc[ids_all].add_prefix("prior_")] + frames, axis=1).astype(np.float32)
Y = d["Y"].loc[ids_all].values; P = d["P"].loc[ids_all].values.astype(bool)
print("X", X.shape, f"{time.time()-t0:.0f}s", flush=True)
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.3, reg_lambda=5.0, n_estimators=350, verbose=-1, random_state=0, n_jobs=8)
def mAP(probs, Yv, Pv):
    probs = np.where(Pv, 0, probs)
    return np.array([average_precision_score(Yv[:, j], probs[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
res = {}
for rep in range(2):
    oof = np.zeros_like(Y, dtype=float)
    for a, b in KFold(5, shuffle=True, random_state=rep).split(X):
        for j in range(40):
            m = ~P[a, j]; y = Y[a, j][m]
            oof[b, j] = lgb.LGBMClassifier(**LGB).fit(X.values[a][m], y).predict_proba(X.values[b])[:, 1]
    ap_all = mAP(oof, Y, P); ap_val = mAP(oof[is_val], Y[is_val], P[is_val]); ap_tr = mAP(oof[~is_val], Y[~is_val], P[~is_val])
    print(f"rep{rep}: 5-fold CV over 3156 -> OOF mAP all={ap_all.mean():.4f} | on val patients={ap_val.mean():.4f} | on train patients={ap_tr.mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    res[rep] = (ap_val, ap_tr)
desc = dict(zip(targets.CODE, targets.DESCRIPTION))
cmp = pd.DataFrame({"cond": [desc[c][:30] for c in codes], "val_pos": Y[is_val].sum(0), "oof_on_val": res[0][0], "oof_on_train": res[0][1]}).round(3)
cmp["gap"] = (cmp.oof_on_val - cmp.oof_on_train).round(3)
pd.set_option("display.width", 200); print(cmp.sort_values("gap").to_string())
