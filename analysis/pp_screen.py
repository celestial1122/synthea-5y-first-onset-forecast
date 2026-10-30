"""Per-target feature screening on the person-period dataset (GPU XGBoost, same folds/outputs as pp_sweep2.py).
For each fold and target: rank the pre-anchor features by univariate AP on the fold's OFFICIAL training rows (at-risk,
cheap: 2.2k rows), keep the top-K plus the always-on set (age, female, prior_*, is_official, yrs_to_end), then fit the d3
booster on all in-fold person-period rows restricted to those columns (one small QuantileDMatrix per target).
screen_gbm.py showed k=200 lifting the official-row LightGBM from 0.207 to 0.213 (terminal 0.415 -> 0.439): feature
dilution is a sample-efficiency lever, so apply it to the best single model. Usage: pp_screen.py 200,400 [--cfg d3]"""
import sys, time, json, argparse, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore"); t0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("ks"); ap.add_argument("--cfg", default="d3"); ap.add_argument("--ds", default="ppds"); ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args(); KS = [int(x) for x in args.ks.split(",")]
HERE = Path(__file__).resolve().parent; DS = HERE / args.ds
import os; EXPORT = (Path(os.environ["EXPORT_DIR"]) / "A") if os.environ.get("EXPORT_DIR") else None; WANDB = bool(os.environ.get("WANDB_PROJECT"))
if EXPORT: EXPORT.mkdir(parents=True, exist_ok=True)
if WANDB: import wandb
CFGS = {"d3": dict(max_depth=3, n=400, eta=0.03, colsample_bytree=0.3, min_child_weight=5), "d6": dict(max_depth=6, n=600, eta=0.03, colsample_bytree=0.3, min_child_weight=5)}
X = pd.read_parquet(DS / "X.parquet").set_index("PATIENT"); M = np.load(DS / "meta.npz", allow_pickle=True)
y, prior, k, split, pid, codes = M["y"], M["prior"].astype(bool), M["k"], M["split"], M["pid"].astype(str), list(M["codes"])
cols = np.array(X.columns); import os; DENSE = bool(os.environ.get("PP_ALWAYS_DENSE"))
always = np.array([i for i, c in enumerate(cols) if c in ("age", "female", "is_official", "yrs_to_end") or c.startswith("prior_") or (DENSE and (c.startswith(("gen_n_", "cost_", "payer_")) or c in ("has_dna", "has_dicom_pre", "logit_pdeath")))])
Xv = np.ascontiguousarray(X.values, dtype=np.float32); del X; gc.collect()
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
off_tr = (k == 0) & (split == "train"); off_va = (k == 0) & (split == "val"); off_te = (k == 0) & (split == "test")
train_ids = pid[off_tr]; Ytr, Ptr = y[off_tr], prior[off_tr]; Yva, Pva = y[off_va], prior[off_va]
p = dict(CFGS[args.cfg]); NR = p.pop("n"); params = dict(objective="binary:logistic", tree_method="hist", device="cuda", subsample=0.8, reg_lambda=5.0, verbosity=0, seed=args.seed, **p)
def uni_ap(Xa, ya):
    n, m = Xa.shape; out = np.zeros(m); npos = ya.sum()
    for sgn in (1, -1):
        order = np.argsort(-sgn * Xa, axis=0, kind="stable"); ys = ya[order]
        cum = np.cumsum(ys, axis=0); prec = cum / np.arange(1, n + 1)[:, None]; out = np.maximum(out, (prec * ys).sum(0) / max(npos, 1))
    return out
print(f"screen {KS} cfg {args.cfg}: rows {len(Xv)}, feats {Xv.shape[1]}, always-on {len(always)} [{time.time()-t0:.0f}s]", flush=True)
if EXPORT: json.dump({"member": "A", "columns": cols.tolist(), "always": always.tolist(), "params": {k_: (v if isinstance(v, (int, float, str)) else str(v)) for k_, v in params.items()}, "rounds": NR, "K": KS, "ds": args.ds}, open(EXPORT / "meta.json", "w"), indent=1)
if WANDB:
    wandb.init(project=os.environ["WANDB_PROJECT"], name=f"memberA_pp_xgb_screen{KS[0]}_{args.ds}{'_s'+str(args.seed) if args.seed else ''}", config=dict(member="A", model="xgboost person-period + per-target screening", cfg=args.cfg, rounds=NR, K=KS, ds=args.ds, seed=args.seed, n_rows=int(len(Xv)), n_feats=int(Xv.shape[1]), **{k_: v for k_, v in params.items() if isinstance(v, (int, float, str)) and k_ != "seed"}))
    wandb.define_metric("round"); wandb.define_metric("curves/*", step_metric="round")
curves = {}
oof = {K: np.zeros((off_tr.sum(), 40)) for K in KS}; pv = {K: np.zeros((off_va.sum(), 40)) for K in KS}; pt = {K: np.zeros((off_va.sum() * 0 + off_te.sum(), 40)) for K in KS}
for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train_ids)):
    held = set(train_ids[b]); in_fold = ~np.isin(pid, list(held)) & ~((k == 0) & (split != "train"))
    rows_a = np.where(off_tr)[0][a]; rows_b = np.where(off_tr)[0][b]; Xf = Xv[in_fold]; yf, pf = y[in_fold], prior[in_fold]
    for j in range(40):
        at_off = ~Ptr[a, j]; ya = Ytr[a, j][at_off]
        if ya.sum() < 5: continue
        u = uni_ap(Xv[rows_a][at_off], ya); u[always] = np.inf; order = np.argsort(-u)
        at = ~pf[:, j]; yy = yf[:, j].astype(np.float32)
        for K in KS:
            sel = np.sort(order[:K + len(always)])
            dtr = xgb.QuantileDMatrix(Xf[:, sel], label=yy, weight=at.astype(np.float32), max_bin=256)
            if WANDB:
                dva = xgb.DMatrix(Xv[off_va][:, sel], label=Yva[:, j].astype(np.float32), weight=(~Pva[:, j]).astype(np.float32)); ev = {}
                bst = xgb.train(params, dtr, num_boost_round=NR, evals=[(dtr, "train"), (dva, "val")], evals_result=ev, verbose_eval=False)
                curves.setdefault((fold, K), []).append((ev["train"]["logloss"], ev["val"]["logloss"]))
            else: bst = xgb.train(params, dtr, num_boost_round=NR)
            if EXPORT and K == KS[0]: bst.save_model(str(EXPORT / f"{codes[j]}_fold{fold}.json")); np.save(EXPORT / f"{codes[j]}_fold{fold}_cols.npy", sel)
            oof[K][b, j] = bst.predict(xgb.DMatrix(Xv[rows_b][:, sel])); pv[K][:, j] += bst.predict(xgb.DMatrix(Xv[off_va][:, sel])) / 5; pt[K][:, j] += bst.predict(xgb.DMatrix(Xv[off_te][:, sel])) / 5
            del dtr
    print(f"  fold {fold} done [{time.time()-t0:.0f}s]", flush=True); del Xf; gc.collect()
    if WANDB and (fold, KS[0]) in curves:
        tr_c = np.mean([c[0] for c in curves[(fold, KS[0])]], 0); va_c = np.mean([c[1] for c in curves[(fold, KS[0])]], 0)
        for r in range(len(tr_c)): wandb.log({"round": r + 1, f"curves/fold{fold}_train_logloss": float(tr_c[r]), f"curves/fold{fold}_val_logloss": float(va_c[r])})
def aps(p_, Yv_, Pv_): p_ = np.where(Pv_, 0, p_); return np.array([average_precision_score(Yv_[:, j], p_[:, j]) if Yv_[:, j].sum() else 0 for j in range(40)])
for K in KS:
    a_ = aps(oof[K], Ytr, Ptr); av = aps(pv[K], Yva, Pva); run = f"screen{K}_{args.cfg}_all{'_dense' if DENSE else ''}{'_s'+str(args.seed) if args.seed else ''}{args.tag}"
    print(f"RESULT {run}: OOF mAP={a_.mean():.4f} | acc {a_[ai].mean():.4f} / term {a_[ti].mean():.4f} / chain {a_[ci].mean():.4f} | val {av.mean():.4f} [{time.time()-t0:.0f}s]")
    np.save(DS / f"{run}_oof.npy", oof[K]); np.save(DS / f"{run}_val.npy", pv[K]); np.save(DS / f"{run}_test.npy", pt[K])
    json.dump({"run": run, "K": K, "oof_by_target": dict(zip(codes, a_.round(4).tolist())), "val_by_target": dict(zip(codes, av.round(4).tolist()))}, open(DS / f"{run}_summary.json", "w"), indent=1)
if WANDB:
    K0 = KS[0]; a_ = aps(oof[K0], Ytr, Ptr); av = aps(pv[K0], Yva, Pva)
    wandb.summary.update({"oof_mAP": float(a_.mean()), "val_mAP": float(av.mean()), "oof_mAP_acc": float(a_[ai].mean()), "oof_mAP_term": float(a_[ti].mean()), "oof_mAP_chain": float(a_[ci].mean()), "val_mAP_acc": float(av[ai].mean()), "val_mAP_term": float(av[ti].mean()), "val_mAP_chain": float(av[ci].mean())})
    wandb.log({"per_target": wandb.Table(columns=["code", "oof_AP", "val_AP"], data=[[c, float(a_[j]), float(av[j])] for j, c in enumerate(codes)])}); wandb.finish()
print(f"done [{time.time()-t0:.0f}s]")
