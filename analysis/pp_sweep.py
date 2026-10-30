"""Person-period model sweep on the cached dataset (pp_build.py). XGBoost on CUDA; grouped 5-fold by patient; OOF on the
official rows of held-out training patients; val/test = fold average. Usage:
    pp_sweep.py <cfg> [--ds ppds] [--targets all|nonterm] [--no-indicator] [--seeds 1] [--wofficial 1.0] [--maxk 99]
Saves <ds>/<run>_{oof,val,test}.npy and prints OOF by target group so runs are comparable with lgb_clean / pp_*."""
import sys, time, json, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore"); t0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("--ds", default="ppds"); ap.add_argument("--targets", default="all"); ap.add_argument("--no-indicator", action="store_true")
ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--wofficial", type=float, default=1.0); ap.add_argument("--maxk", type=int, default=99); ap.add_argument("--tag", default="")
args = ap.parse_args()
HERE = Path(__file__).resolve().parent; DS = HERE / args.ds
CFGS = {
    "d3":      dict(max_depth=3, n_estimators=400, learning_rate=0.03, colsample_bytree=0.3, min_child_weight=5),
    "d4":      dict(max_depth=4, n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, min_child_weight=5),
    "d6":      dict(max_depth=6, n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, min_child_weight=5),
    "d4slow":  dict(max_depth=4, n_estimators=1200, learning_rate=0.015, colsample_bytree=0.5, min_child_weight=5),
    "d4mcw20": dict(max_depth=4, n_estimators=600, learning_rate=0.03, colsample_bytree=0.3, min_child_weight=20),
    "d5cs2":   dict(max_depth=5, n_estimators=800, learning_rate=0.03, colsample_bytree=0.2, min_child_weight=10),
    "d3n800":  dict(max_depth=3, n_estimators=800, learning_rate=0.03, colsample_bytree=0.3, min_child_weight=5),
    "d8":      dict(max_depth=8, n_estimators=500, learning_rate=0.03, colsample_bytree=0.3, min_child_weight=10, gamma=0.5),
}
cfg = dict(device="cuda", tree_method="hist", subsample=0.8, reg_lambda=5.0, verbosity=0, **CFGS[args.cfg])
X = pd.read_parquet(DS / "X.parquet").set_index("PATIENT"); M = np.load(DS / "meta.npz", allow_pickle=True)
y, prior, k, split, pid, codes = M["y"], M["prior"].astype(bool), M["k"], M["split"], M["pid"].astype(str), list(M["codes"])
cols = list(X.columns)
if args.no_indicator: X = X.drop(columns=["is_official", "yrs_to_end"])
keep = (k <= args.maxk); X, y, prior, k, split, pid = X[keep], y[keep], prior[keep], k[keep], split[keep], pid[keep]
Xv = X.values
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
tj = list(range(40)) if args.targets == "all" else [j for j in range(40) if j not in ti]
off_tr = (k == 0) & (split == "train"); off_va = (k == 0) & (split == "val"); off_te = (k == 0) & (split == "test")
train_ids = pid[off_tr]; Ytr, Ptr = y[off_tr], prior[off_tr]; Yva, Pva = y[off_va], prior[off_va]
print(f"{args.cfg}: rows {len(X)} (train landmark rows {int((k>0)&(split=='train')).sum() if False else int(((k>0)&(split=='train')).sum())}, transductive rows {int(((k>0)&(split!='train')).sum())}), feats {Xv.shape[1]}, targets {len(tj)}, indicator={not args.no_indicator} [{time.time()-t0:.0f}s]", flush=True)
oof = np.zeros((off_tr.sum(), 40)); pv = np.zeros((off_va.sum(), 40)); pt = np.zeros((off_te.sum(), 40))
pos_tr = {j: 0 for j in tj}
for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train_ids)):
    held = set(train_ids[b]); in_fold = ~np.isin(pid, list(held)) & ~((k == 0) & (split != "train"))   # all rows of non-held patients (incl. transductive rows if present)
    Xf, yf, pf, kf = Xv[in_fold], y[in_fold], prior[in_fold], k[in_fold]; wf = np.where(kf == 0, args.wofficial, 1.0)
    rows_b = np.where(off_tr)[0][b]
    for j in tj:
        at = ~pf[:, j]; yy = yf[at, j]
        if yy.sum() < 5: continue
        for s in range(args.seeds):
            mdl = xgb.XGBClassifier(random_state=s, **cfg).fit(Xf[at], yy, sample_weight=wf[at])
            oof[b, j] += mdl.predict_proba(Xv[rows_b])[:, 1] / args.seeds; pv[:, j] += mdl.predict_proba(Xv[off_va])[:, 1] / (5 * args.seeds); pt[:, j] += mdl.predict_proba(Xv[off_te])[:, 1] / (5 * args.seeds)
    print(f"  fold {fold} done [{time.time()-t0:.0f}s]", flush=True)
def aps(p, Yv, Pv): p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
a = aps(oof, Ytr, Ptr); av = aps(pv, Yva, Pva)
run = f"{args.cfg}_{args.targets}{'_noind' if args.no_indicator else ''}{'_s'+str(args.seeds) if args.seeds>1 else ''}{'_w'+str(args.wofficial) if args.wofficial!=1 else ''}{'_k'+str(args.maxk) if args.maxk<99 else ''}{args.tag}"
sub = lambda idx: float(np.mean(a[[j for j in idx if j in tj]])) if any(j in tj for j in idx) else float("nan")
print(f"RESULT {run}: OOF mAP(targets run)={a[tj].mean():.4f} | acc {sub(ai):.4f} / term {sub(ti):.4f} / chain {sub(ci):.4f} | val mAP(targets run)={av[tj].mean():.4f} [{time.time()-t0:.0f}s]")
np.save(DS / f"{run}_oof.npy", oof); np.save(DS / f"{run}_val.npy", pv); np.save(DS / f"{run}_test.npy", pt)
json.dump({"run": run, "cfg": cfg, "oof_by_target": dict(zip(codes, a.round(4).tolist())), "val_by_target": dict(zip(codes, av.round(4).tolist()))}, open(DS / f"{run}_summary.json", "w"), indent=1)
