"""Fast person-period sweep: one GPU QuantileDMatrix per fold shared by all configs x targets (at-risk masking via zero
sample weights instead of row subsetting), native xgb.train API. Same folds / outputs as pp_sweep.py.
Usage: pp_sweep2.py d3,d4,d6 [--ds ppds] [--targets all|nonterm] [--no-indicator] [--seeds 1] [--wofficial 1.0] [--tag x]"""
import sys, time, json, argparse, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore"); t0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("cfgs"); ap.add_argument("--ds", default="ppds"); ap.add_argument("--targets", default="all"); ap.add_argument("--no-indicator", action="store_true")
ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--wofficial", type=float, default=1.0); ap.add_argument("--tag", default=""); ap.add_argument("--maxk", type=int, default=99); ap.add_argument("--block", default="all"); ap.add_argument("--frac", type=float, default=1.0)
args = ap.parse_args()
HERE = Path(__file__).resolve().parent; DS = HERE / args.ds
CFGS = {
    "d3":      dict(max_depth=3, n=400,  eta=0.03,  colsample_bytree=0.3, min_child_weight=5),
    "d4":      dict(max_depth=4, n=600,  eta=0.03,  colsample_bytree=0.3, min_child_weight=5),
    "d6":      dict(max_depth=6, n=600,  eta=0.03,  colsample_bytree=0.3, min_child_weight=5),
    "d4slow":  dict(max_depth=4, n=1200, eta=0.015, colsample_bytree=0.5, min_child_weight=5),
    "d4mcw20": dict(max_depth=4, n=600,  eta=0.03,  colsample_bytree=0.3, min_child_weight=20),
    "d5cs2":   dict(max_depth=5, n=800,  eta=0.03,  colsample_bytree=0.2, min_child_weight=10),
    "d3n800":  dict(max_depth=3, n=800,  eta=0.03,  colsample_bytree=0.3, min_child_weight=5),
    "d3n200":  dict(max_depth=3, n=200,  eta=0.03,  colsample_bytree=0.3, min_child_weight=5),
    "d3spw3":  dict(max_depth=3, n=400,  eta=0.03,  colsample_bytree=0.3, min_child_weight=5, scale_pos_weight=3),
    "d3cs6":   dict(max_depth=3, n=400,  eta=0.03,  colsample_bytree=0.6, min_child_weight=5),
    "d8":      dict(max_depth=8, n=500,  eta=0.03,  colsample_bytree=0.3, min_child_weight=10, gamma=0.5),
    "d4n300":  dict(max_depth=4, n=300,  eta=0.05,  colsample_bytree=0.3, min_child_weight=5),
    "d2n800":  dict(max_depth=2, n=800,  eta=0.05,  colsample_bytree=0.3, min_child_weight=5),
    "lw16":    dict(max_depth=0, n=600,  eta=0.03,  colsample_bytree=0.3, min_child_weight=5, grow_policy="lossguide", max_leaves=16),
    "lw32":    dict(max_depth=0, n=800,  eta=0.03,  colsample_bytree=0.3, min_child_weight=10, grow_policy="lossguide", max_leaves=32),
}
cfg_names = args.cfgs.split(",")
X = pd.read_parquet(DS / "X.parquet").set_index("PATIENT"); M = np.load(DS / "meta.npz", allow_pickle=True)
y, prior, k, split, pid, codes = M["y"], M["prior"].astype(bool), M["k"], M["split"], M["pid"].astype(str), list(M["codes"])
if args.no_indicator: X = X.drop(columns=["is_official", "yrs_to_end"])
def block_cols(cols, block):
    """Nested feature blocks (feature-dilution control): S1 static+prior, S2 +numeric summaries, S3 +COND/MED tokens, all."""
    tok = lambda c: any(c.startswith(p) for p in ("cnt_", "c5_", "c1_", "rec_", "first_"))
    numr = lambda c: any(c.startswith(p) for p in ("last_", "max_", "min_", "m5_", "std_", "nobs_", "trend_", "nrec_"))
    static = [c for c in cols if not tok(c) and not numr(c)]
    if block == "S1": return static
    if block == "S2": return static + [c for c in cols if numr(c)]
    if block == "S3": return static + [c for c in cols if numr(c) or (tok(c) and (":COND:" in ("_" + c) or "COND:" in c or "MED:" in c))]
    return list(cols)
if args.block != "all": X = X[block_cols(list(X.columns), args.block)]
keep = (k <= args.maxk); Xv = np.ascontiguousarray(X.values[keep], dtype=np.float32); y, prior, k, split, pid = y[keep], prior[keep], k[keep], split[keep], pid[keep]; del X; gc.collect()
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
tj = list(range(40)) if args.targets == "all" else [j for j in range(40) if j not in ti]
off_tr = (k == 0) & (split == "train"); off_va = (k == 0) & (split == "val"); off_te = (k == 0) & (split == "test")
train_ids = pid[off_tr]; Ytr, Ptr = y[off_tr], prior[off_tr]; Yva, Pva = y[off_va], prior[off_va]
dva = xgb.DMatrix(Xv[off_va]); dte = xgb.DMatrix(Xv[off_te])
print(f"configs {cfg_names}: rows {len(Xv)} (landmark rows {int((k>0).sum())}, transductive {int(((k>0)&(split!='train')).sum())}), feats {Xv.shape[1]}, targets {len(tj)}, indicator={not args.no_indicator} [{time.time()-t0:.0f}s]", flush=True)
oof = {c: np.zeros((off_tr.sum(), 40)) for c in cfg_names}; pv = {c: np.zeros((off_va.sum(), 40)) for c in cfg_names}; pt = {c: np.zeros((off_te.sum(), 40)) for c in cfg_names}
for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train_ids)):
    held = set(train_ids[b]); in_fold = ~np.isin(pid, list(held)) & ~((k == 0) & (split != "train"))
    if args.frac < 1: in_fold &= np.isin(pid, np.random.RandomState(100 + fold).choice(train_ids[a], int(round(args.frac * len(a))), replace=False))  # learning curve: subsample fold-training patients
    Xf = Xv[in_fold]; yf, pf, kf = y[in_fold], prior[in_fold], k[in_fold]; wbase = np.where(kf == 0, args.wofficial, 1.0).astype(np.float32)
    dtr = xgb.QuantileDMatrix(Xf, max_bin=256); rows_b = np.where(off_tr)[0][b]; db = xgb.DMatrix(Xv[rows_b]); del Xf; gc.collect()
    print(f"  fold {fold}: DMatrix built ({int(in_fold.sum())} rows) [{time.time()-t0:.0f}s]", flush=True)
    for j in tj:
        at = ~pf[:, j]; yy = yf[:, j].astype(np.float32)
        if yy[at].sum() < 5: continue
        dtr.set_label(yy); dtr.set_weight(wbase * at)
        for c in cfg_names:
            p = dict(CFGS[c]); n = p.pop("n"); params = dict(objective="binary:logistic", tree_method="hist", device="cuda", subsample=0.8, reg_lambda=5.0, verbosity=0, **p)
            for s in range(args.seeds):
                params["seed"] = s; bst = xgb.train(params, dtr, num_boost_round=n)
                oof[c][b, j] += bst.predict(db) / args.seeds; pv[c][:, j] += bst.predict(dva) / (5 * args.seeds); pt[c][:, j] += bst.predict(dte) / (5 * args.seeds)
    print(f"  fold {fold} done [{time.time()-t0:.0f}s]", flush=True)
    del dtr, db; gc.collect()
def aps(p, Yv, Pv): p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
sfx = f"_{args.targets}{'_noind' if args.no_indicator else ''}{'_s'+str(args.seeds) if args.seeds>1 else ''}{'_w'+str(args.wofficial) if args.wofficial!=1 else ''}{'_k'+str(args.maxk) if args.maxk<99 else ''}{'_'+args.block if args.block!='all' else ''}{'_f'+str(args.frac) if args.frac<1 else ''}{args.tag}"
for c in cfg_names:
    a = aps(oof[c], Ytr, Ptr); av = aps(pv[c], Yva, Pva); run = c + sfx
    sub = lambda idx: float(np.mean(a[[j for j in idx if j in tj]])) if any(j in tj for j in idx) else float("nan")
    print(f"RESULT {run}: OOF mAP(targets run)={a[tj].mean():.4f} | acc {sub(ai):.4f} / term {sub(ti):.4f} / chain {sub(ci):.4f} | val {av[tj].mean():.4f}")
    np.save(DS / f"{run}_oof.npy", oof[c]); np.save(DS / f"{run}_val.npy", pv[c]); np.save(DS / f"{run}_test.npy", pt[c])
    json.dump({"run": run, "cfg": CFGS[c], "oof_by_target": dict(zip(codes, a.round(4).tolist())), "val_by_target": dict(zip(codes, av.round(4).tolist()))}, open(DS / f"{run}_summary.json", "w"), indent=1)
print(f"done [{time.time()-t0:.0f}s]")
