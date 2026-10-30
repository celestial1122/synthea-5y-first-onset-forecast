"""CPU LightGBM twin of pp_sweep2.py (same cached person-period dataset, same patient folds, same outputs) so that the
feature-block question can be answered while the GPU is busy. Count-based leaf constraint (min_child_samples) instead of
XGBoost's Hessian-based one. Usage: pp_sweep_lgb.py <cfg> [--ds ppds] [--block S1|S2|S3|all] [--targets all|nonterm] [--no-indicator]"""
import time, json, argparse, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore"); t0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("--ds", default="ppds"); ap.add_argument("--block", default="all"); ap.add_argument("--targets", default="all"); ap.add_argument("--no-indicator", action="store_true"); ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
HERE = Path(__file__).resolve().parent; DS = HERE / args.ds; OUT = HERE / "pplgb"; OUT.mkdir(exist_ok=True)
CFGS = {"nl8":  dict(num_leaves=8,  min_child_samples=15, n_estimators=500, learning_rate=0.03, colsample_bytree=0.3),
        "nl16": dict(num_leaves=16, min_child_samples=15, n_estimators=600, learning_rate=0.03, colsample_bytree=0.3),
        "nl8s": dict(num_leaves=8,  min_child_samples=15, n_estimators=500, learning_rate=0.03, colsample_bytree=0.6),
        "nl4":  dict(num_leaves=4,  min_child_samples=15, n_estimators=700, learning_rate=0.03, colsample_bytree=0.5)}
cfg = dict(objective="binary", verbose=-1, n_jobs=12, subsample=0.8, subsample_freq=1, reg_lambda=5.0, random_state=args.seed, **CFGS[args.cfg])
X = pd.read_parquet(DS / "X.parquet").set_index("PATIENT"); M = np.load(DS / "meta.npz", allow_pickle=True)
y, prior, k, split, pid, codes = M["y"], M["prior"].astype(bool), M["k"], M["split"], M["pid"].astype(str), list(M["codes"])
if args.no_indicator: X = X.drop(columns=["is_official", "yrs_to_end"])
tok = lambda c: any(c.startswith(p) for p in ("cnt_", "c5_", "c1_", "rec_", "first_")); numr = lambda c: any(c.startswith(p) for p in ("last_", "max_", "min_", "m5_", "std_", "nobs_", "trend_", "nrec_"))
cols = list(X.columns); static = [c for c in cols if not tok(c) and not numr(c)]
sel = {"S1": static, "S2": static + [c for c in cols if numr(c)], "S3": static + [c for c in cols if numr(c) or (tok(c) and ("COND:" in c or "MED:" in c))], "all": cols}[args.block]
Xv = np.ascontiguousarray(X[sel].values, dtype=np.float32); del X; gc.collect()
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
tj = list(range(40)) if args.targets == "all" else [j for j in range(40) if j not in ti]
off_tr = (k == 0) & (split == "train"); off_va = (k == 0) & (split == "val"); off_te = (k == 0) & (split == "test")
train_ids = pid[off_tr]; Ytr, Ptr = y[off_tr], prior[off_tr]; Yva, Pva = y[off_va], prior[off_va]
print(f"{args.cfg} block {args.block}: rows {len(Xv)}, feats {Xv.shape[1]}, targets {len(tj)} [{time.time()-t0:.0f}s]", flush=True)
oof = np.zeros((off_tr.sum(), 40)); pv = np.zeros((off_va.sum(), 40)); pt = np.zeros((off_te.sum(), 40))
for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train_ids)):
    held = set(train_ids[b]); in_fold = ~np.isin(pid, list(held)) & ~((k == 0) & (split != "train"))
    Xf, yf, pf = Xv[in_fold], y[in_fold], prior[in_fold]; rows_b = np.where(off_tr)[0][b]
    for j in tj:
        at = ~pf[:, j]; yy = yf[at, j]
        if yy.sum() < 5: continue
        mdl = lgb.LGBMClassifier(**cfg).fit(Xf[at], yy)
        oof[b, j] = mdl.predict_proba(Xv[rows_b])[:, 1]; pv[:, j] += mdl.predict_proba(Xv[off_va])[:, 1] / 5; pt[:, j] += mdl.predict_proba(Xv[off_te])[:, 1] / 5
    print(f"  fold {fold} done [{time.time()-t0:.0f}s]", flush=True)
def aps(p, Yv, Pv): p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
a = aps(oof, Ytr, Ptr); av = aps(pv, Yva, Pva); run = f"{args.cfg}_{args.block}_{args.targets}{'_noind' if args.no_indicator else ''}{'_seed'+str(args.seed) if args.seed else ''}{args.tag}"
sub = lambda idx: float(np.mean(a[[j for j in idx if j in tj]])) if any(j in tj for j in idx) else float("nan")
print(f"RESULT {run}: OOF mAP(targets run)={a[tj].mean():.4f} | acc {sub(ai):.4f} / term {sub(ti):.4f} / chain {sub(ci):.4f} | val {av[tj].mean():.4f} [{time.time()-t0:.0f}s]")
np.save(OUT / f"{run}_oof.npy", oof); np.save(OUT / f"{run}_val.npy", pv); np.save(OUT / f"{run}_test.npy", pt)
json.dump({"run": run, "cfg": cfg, "oof_by_target": dict(zip(codes, a.round(4).tolist()))}, open(OUT / f"{run}_summary.json", "w"), indent=1)
