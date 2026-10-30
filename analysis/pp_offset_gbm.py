"""Offset boosting: official-window GBM started from the person-period hazard model.
Classic hazard-regression trick (offset term): log-odds = logit(pp_hat(x)) + f(x), where pp_hat is the natural 5-year risk
learned from all pre-anchor person-years (cross-fitted, so for every training patient it comes from models that never saw
them) and f is a small LightGBM fitted on the 2,791 official rows with init_score = logit(pp_hat). f only has to learn the
*correction* from natural windows to the official (record-end-aligned) window - the death/era coupling - instead of the
whole hazard, which is what the 2,791 rows can afford.
Usage: pp_offset_gbm.py <pp_dir> <pp_stem> [--targets nonterm|all] [--trees 150] [--lr 0.03] [--leaves 4] [--tag x]
Writes offset/<tag>_{oof,val,test}.npy (targets not run are copied from lgb_clean so the file is a full 40-column candidate)."""
import sys, time, json, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("pp_dir"); ap.add_argument("pp_stem"); ap.add_argument("--targets", default="nonterm"); ap.add_argument("--trees", type=int, default=150)
ap.add_argument("--lr", type=float, default=0.03); ap.add_argument("--leaves", type=int, default=4); ap.add_argument("--mcs", type=int, default=15); ap.add_argument("--tag", default=None); ap.add_argument("--reps", type=int, default=2)
args = ap.parse_args(); tag = args.tag or f"{args.pp_dir}_{args.pp_stem.strip('_')}_{args.targets}_t{args.trees}_l{args.leaves}"
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "offset"; OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
X = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(pats.PATIENT)
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Xtr, Xva, Xte = X.values[tr], X.values[va], X.values[te]; Ytr, Ptr, Yva, Pva = Y.values[tr], P.values[tr].astype(bool), Y.values[va], P.values[va].astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
tj = list(range(40)) if args.targets == "all" else [j for j in range(40) if j not in ti]
pp = {s: np.load(HERE / args.pp_dir / f"{args.pp_stem}{s}.npy") for s in ("oof", "val", "test")}
base = {s: np.load(HERE / "clean" / f"{s}.npy") for s in ("oof", "val", "test")}
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5))); sg = lambda z: 1 / (1 + np.exp(-z))
cfg = dict(objective="binary", verbose=-1, n_jobs=10, learning_rate=args.lr, n_estimators=args.trees, num_leaves=args.leaves, min_child_samples=args.mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0)
oof = base["oof"].copy(); pv = base["val"].copy(); pt = base["test"].copy(); oof[:, tj] = 0; pv[:, tj] = 0; pt[:, tj] = 0
for r in range(args.reps):
    for a, b in KFold(5, shuffle=True, random_state=r).split(Xtr):
        for j in tj:
            m = ~Ptr[a, j]; off_a = lg(pp["oof"][a, j]); y = Ytr[a, j]
            if y[m].sum() < 5: continue
            mdl = lgb.LGBMClassifier(random_state=r, **cfg).fit(Xtr[a][m], y[m], init_score=off_a[m])
            oof[b, j] += sg(mdl.predict(Xtr[b], raw_score=True) + lg(pp["oof"][b, j])) / args.reps
            pv[:, j] += sg(mdl.predict(Xva, raw_score=True) + lg(pp["val"][:, j])) / (5 * args.reps); pt[:, j] += sg(mdl.predict(Xte, raw_score=True) + lg(pp["test"][:, j])) / (5 * args.reps)
    print(f"  rep {r} done [{time.time()-t0:.0f}s]", flush=True)
def aps(p, Yv, Pv): p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
a = aps(oof, Ytr, Ptr); a_pp = aps(pp["oof"], Ytr, Ptr); a_base = aps(base["oof"], Ytr, Ptr); av = aps(pv, Yva, Pva)
sub = lambda arr, idx: float(np.mean(arr[[j for j in idx if j in tj]])) if any(j in tj for j in idx) else float("nan")
print(f"RESULT offset {tag}: OOF mAP(targets run)={a[tj].mean():.4f} [pp alone {a_pp[tj].mean():.4f}, lgb_clean {a_base[tj].mean():.4f}] | acc {sub(a, ai):.4f} (pp {sub(a_pp, ai):.4f}) / term {sub(a, ti):.4f} (pp {sub(a_pp, ti):.4f}) / chain {sub(a, ci):.4f} (pp {sub(a_pp, ci):.4f}) | val {av[tj].mean():.4f} | all-40 OOF {a.mean():.4f} [{time.time()-t0:.0f}s]")
print(pd.DataFrame({"cond": [desc[codes[j]][:30] for j in tj], "offset": a[tj].round(3), "pp": a_pp[tj].round(3), "lgb_clean": a_base[tj].round(3)}).to_string())
np.save(OUT / f"{tag}_oof.npy", oof); np.save(OUT / f"{tag}_val.npy", pv); np.save(OUT / f"{tag}_test.npy", pt)
