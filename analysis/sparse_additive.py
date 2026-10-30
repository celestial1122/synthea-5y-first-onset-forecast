"""Inductive-bias test: sparse ADDITIVE models (matching independent Synthea modules with per-state constant hazards)
vs the depth-8 GBM, on the clean dataset, 5-fold x 2 OOF.
  (i) L1 logistic regression (saga) on standardized features + missing indicators, C in {0.05, 0.2}
  (ii) boosted stumps (num_leaves=2, 1500 rounds, lr 0.02, bagging) = purely additive gradient boosting
"""
import time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "v4"; OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
X = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(pats.PATIENT)
num_cols = [c for c in X.columns if c.split("_")[0] in ("last", "max", "min", "m5", "std", "trend", "rec", "first")]
Xl = X.copy(); miss = Xl[num_cols].isna().astype(np.float32).add_prefix("miss_"); Xl = pd.concat([Xl.fillna(0), miss], axis=1)
cnt_like = [c for c in Xl.columns if c.split("_")[0] in ("cnt", "c5", "c1", "nobs", "n")]; Xl[cnt_like] = np.log1p(Xl[cnt_like])
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Ytr, Ptr, Yva, Pva = Y.values[tr], P.values[tr].astype(bool), Y.values[va], P.values[va].astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
def aps(p, Yv, Pv):
    p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
folds = [(r, a, b) for r in range(2) for a, b in KFold(5, shuffle=True, random_state=r).split(np.arange(tr.sum()))]
def cv(fit_predict, Xm):
    Xtr, Xva, Xte = Xm.values[tr], Xm.values[va], Xm.values[te]
    oof = np.zeros((len(Xtr), 40)); pv = np.zeros((len(Xva), 40)); pt = np.zeros((len(Xte), 40))
    for r, a, b in folds:
        for j in range(40):
            m = ~Ptr[a, j]; y = Ytr[a, j][m]
            po, pvv, ptt = fit_predict(Xtr[a][m], y, Xtr[b], Xva, Xte)
            oof[b, j] += po / 2; pv[:, j] += pvv / len(folds); pt[:, j] += ptt / len(folds)
    return oof, pv, pt
def l1_factory(C):
    def fp(Xa, y, Xb, Xv, Xt):
        mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
        f = lambda Z: np.clip((Z - mu) / sd, -6, 6)
        m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=3000, tol=1e-3).fit(f(Xa), y)
        return m.predict_proba(f(Xb))[:, 1], m.predict_proba(f(Xv))[:, 1], m.predict_proba(f(Xt))[:, 1]
    return fp
def stumps(Xa, y, Xb, Xv, Xt):
    m = lgb.LGBMClassifier(objective="binary", verbose=-1, n_jobs=8, num_leaves=2, learning_rate=0.02, n_estimators=1500, subsample=0.7, subsample_freq=1, colsample_bytree=0.3, min_child_samples=15, reg_lambda=5.0).fit(Xa, y)
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
res = {}
for name, fp, Xm in (("L1 logistic C=0.05", l1_factory(0.05), Xl), ("L1 logistic C=0.2", l1_factory(0.2), Xl), ("boosted stumps 1500", stumps, X)):
    oof, pv, pt = cv(fp, Xm); a = aps(oof, Ytr, Ptr); v = aps(pv, Yva, Pva); res[name] = (oof, pv, pt, a, v)
    print(f"{name:24s} OOF mAP={a.mean():.4f} (acc {a[ai].mean():.3f}/term {a[ti].mean():.3f}/chain {a[ci].mean():.3f}) | val {v.mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    np.save(OUT / f"{name.replace(' ', '_').replace('=', '')}_oof.npy", oof); np.save(OUT / f"{name.replace(' ', '_').replace('=', '')}_val.npy", pv); np.save(OUT / f"{name.replace(' ', '_').replace('=', '')}_test.npy", pt)
g = np.load(HERE / "clean" / "oof.npy"); gv = np.load(HERE / "clean" / "val.npy")
print(f"{'clean GBM (reference)':24s} OOF mAP={aps(g, Ytr, Ptr).mean():.4f} | val {aps(gv, Yva, Pva).mean():.4f}")
for name in res:
    o = (res[name][0] + g) / 2; print(f"{'avg GBM + ' + name:36s} OOF mAP={aps(o, Ytr, Ptr).mean():.4f} | val {aps((res[name][1] + gv) / 2, Yva, Pva).mean():.4f}")
per = pd.DataFrame({"cond": [desc[c][:32] for c in codes], "gbm": aps(g, Ytr, Ptr), **{k: v[3] for k, v in res.items()}}, index=codes).round(3)
pd.set_option("display.width", 220); print(per.to_string()); per.to_csv(OUT / "per_condition_additive.csv")
json.dump({k: {"oof": float(v[3].mean()), "val": float(v[4].mean())} for k, v in res.items()}, open(OUT / "summary_additive.json", "w"), indent=1)
