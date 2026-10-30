"""Feature-dilution control on the official rows: LightGBM on nested feature blocks of the clean dataset
(S1 = static + prior indicators, S2 = S1 + numeric summaries, S3 = S2 + COND/MED tokens; 'all' = the 2317-column baseline).
The mechanism probe showed hand-picked subsets beating the full-feature GBM on pregnancy / Alzheimer / osteoporosis /
anemia at the official anchor, i.e. with 30-150 positives the 2,000 irrelevant columns cost real AP. 5-fold x 2 reps;
outputs blocks/{S1,S2,S3}_{oof,val,test}.npy as ensemble candidates."""
import time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; import genfeat; OUT = genfeat.outdir("blocks"); OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
X = genfeat.augment(pd.read_parquet(HERE / "clean" / "X.parquet")).reindex(pats.PATIENT); cols = list(X.columns)
tok = lambda c: any(c.startswith(p) for p in ("cnt_", "c5_", "c1_", "rec_", "first_"))
numr = lambda c: any(c.startswith(p) for p in ("last_", "max_", "min_", "m5_", "std_", "nobs_", "trend_"))
static = [c for c in cols if not tok(c) and not numr(c) and not c.startswith("active_")]
BLOCKS = {"S1": static, "S2": static + [c for c in cols if numr(c)], "S3": static + [c for c in cols if numr(c) or c.startswith("active_") or (tok(c) and ("COND:" in c or "MED:" in c))]}
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Ytr, Ptr, Yva, Pva = Y.values[tr], P.values[tr].astype(bool), Y.values[va], P.values[va].astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
def aps(p, Yv, Pv): p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
ref = aps(np.load(HERE / "clean" / "oof.npy"), Ytr, Ptr)
cfg = dict(objective="binary", verbose=-1, n_jobs=12, learning_rate=0.03, n_estimators=350, subsample=0.8, subsample_freq=1, reg_lambda=5.0, num_leaves=8, min_child_samples=15)
res = {}
for name, sub in BLOCKS.items():
    cs = 0.8 if len(sub) < 100 else 0.5 if len(sub) < 500 else 0.3
    Xtr, Xva, Xte = X[sub].values[tr], X[sub].values[va], X[sub].values[te]
    oof = np.zeros((len(Xtr), 40)); pv = np.zeros((len(Xva), 40)); pt = np.zeros((len(Xte), 40))
    for r in range(2):
        for a, b in KFold(5, shuffle=True, random_state=r).split(Xtr):
            for j in range(40):
                m = ~Ptr[a, j]; mdl = lgb.LGBMClassifier(random_state=r, colsample_bytree=cs, **cfg).fit(Xtr[a][m], Ytr[a, j][m])
                oof[b, j] += mdl.predict_proba(Xtr[b])[:, 1] / 2; pv[:, j] += mdl.predict_proba(Xva)[:, 1] / 10; pt[:, j] += mdl.predict_proba(Xte)[:, 1] / 10
    a = aps(oof, Ytr, Ptr); av = aps(pv, Yva, Pva); res[name] = a
    print(f"RESULT block {name} ({len(sub)} feats): OOF mAP={a.mean():.4f} (acc {a[ai].mean():.3f} / term {a[ti].mean():.3f} / chain {a[ci].mean():.3f}) | val {av.mean():.4f} | full-feature lgb_clean OOF {ref.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
    np.save(OUT / f"{name}_oof.npy", oof); np.save(OUT / f"{name}_val.npy", pv); np.save(OUT / f"{name}_test.npy", pt)
per = pd.DataFrame({"cond": [desc[c][:32] for c in codes], **{k: v.round(3) for k, v in res.items()}, "all": ref.round(3)}, index=codes)
per["best_block"] = per[list(BLOCKS) + ["all"]].idxmax(1); pd.set_option("display.width", 200); print(per.to_string()); per.to_csv(OUT / "per_condition_blocks.csv")
print("per-target oracle over blocks+all:", round(float(per[list(BLOCKS) + ['all']].max(1).mean()), 4))
