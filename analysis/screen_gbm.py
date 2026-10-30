"""Data-driven feature-dilution control: per target and per fold, keep only the top-k pre-anchor features by univariate
AP on the fold's at-risk training rows (plus age/female/prior_* always), then fit the lgb_clean LightGBM on that subset.
The manual S1/S2/S3 blocks showed 64 static features beating 2,317 on osteoporosis / Alzheimer / pregnancy; this tests
whether a per-target screen generalises that. Usage: screen_gbm.py 60,200   -> screen/k{K}_{oof,val,test}.npy"""
import sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; import genfeat; OUT = genfeat.outdir("screen"); OUT.mkdir(exist_ok=True)
import os; EXPORT = (Path(os.environ["EXPORT_DIR"]) / "B") if os.environ.get("EXPORT_DIR") else None; WANDB = bool(os.environ.get("WANDB_PROJECT"))
if EXPORT: EXPORT.mkdir(parents=True, exist_ok=True)
if WANDB: import wandb
KS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "60,200").split(",")]
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
Xall = genfeat.augment(pd.read_parquet(HERE / "clean" / "X.parquet")).reindex(pats.PATIENT); cols = np.array(Xall.columns)
always = np.array([i for i, c in enumerate(cols) if c in ("age", "female") or c.startswith("prior_")])
S = {s: (pats.split == s).values for s in ("train", "val", "test")}
X = {s: Xall.values[S[s]].astype(np.float32) for s in S}; Yv = {s: Y.values[S[s]] for s in S}; Pv = {s: P.values[S[s]].astype(bool) for s in S}
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
import os; LGB_N = int(os.environ.get("LGB_N", "350")); LGB_SEED = int(os.environ.get("LGB_SEED", "0")); GRP_N = os.environ.get("LGB_N_GROUP", "")
NSFX = (f"_n{LGB_N}" if LGB_N != 350 else "") + ("_grp" if GRP_N else "") + (f"_s{LGB_SEED}" if LGB_SEED else "")
GRP_ROUNDS = {k: int(v) for k, v in (kv.split(":") for kv in GRP_N.split(","))} if GRP_N else {}
TERM_ = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACC_ = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
def rounds_for(code): return GRP_ROUNDS.get("term" if code in TERM_ else "acc" if code in ACC_ else "chain", LGB_N) if GRP_ROUNDS else LGB_N
cfg = dict(objective="binary", verbose=-1, n_jobs=12, learning_rate=0.03, subsample=0.8, subsample_freq=1, reg_lambda=5.0, num_leaves=8, min_child_samples=15, random_state=LGB_SEED)
folds = list(KFold(5, shuffle=True, random_state=0).split(X["train"]))
def aps(p, s="train"): p = np.where(Pv[s], 0, p); return np.array([average_precision_score(Yv[s][:, j], p[:, j]) if Yv[s][:, j].sum() else 0 for j in range(40)])
def uni_ap(Xa, ya):
    """univariate AP of each column (max of the column and its negation), vectorised by ranking"""
    n, m = Xa.shape; out = np.zeros(m); npos = ya.sum()
    for sgn in (1, -1):
        order = np.argsort(-sgn * Xa, axis=0, kind="stable"); ys = ya[order]                  # n x m, labels in score order
        cum = np.cumsum(ys, axis=0); prec = cum / np.arange(1, n + 1)[:, None]; ap = (prec * ys).sum(0) / max(npos, 1)
        out = np.maximum(out, ap)
    return out
res = {}
if EXPORT: json.dump({"member": "B", "columns": cols.tolist(), "always": always.tolist(), "cfg": cfg, "rounds": LGB_N, "group_rounds": GRP_ROUNDS, "K": KS, "genomics_mode": os.environ.get("EHR_GENOMICS", "")}, open(EXPORT / "meta.json", "w"), indent=1)
if WANDB:
    wandb.init(project=os.environ["WANDB_PROJECT"], name=f"memberB_lgb_official_screen{KS[0]}{NSFX}", config=dict(member="B", model="lightgbm official rows + per-target screening", rounds=LGB_N, group_rounds=GRP_ROUNDS, K=KS, seed=LGB_SEED, n_rows=int(len(X["train"])), n_feats=int(X["train"].shape[1]), **cfg))
    wandb.define_metric("round"); wandb.define_metric("curves/*", step_metric="round")
for K in KS:
    oof = np.zeros((len(X["train"]), 40)); pv = np.zeros((len(X["val"]), 40)); pt = np.zeros((len(X["test"]), 40)); nsel = []
    for fi, (a, b) in enumerate(folds):
        cur = []
        for j in range(40):
            at = ~Pv["train"][a, j]; Xa = X["train"][a][at]; ya = Yv["train"][a, j][at]
            if ya.sum() < 5: continue
            u = uni_ap(Xa, ya); u[always] = np.inf; sel = np.argsort(-u)[:K + len(always)]; nsel.append(len(sel))
            cs = 0.8 if len(sel) < 100 else 0.5 if len(sel) < 500 else 0.3
            fit_kw = dict(eval_set=[(Xa[:, sel], ya), (X["val"][:, sel], Yv["val"][:, j])], eval_names=["train", "val"], eval_metric="binary_logloss") if WANDB else {}
            m = lgb.LGBMClassifier(colsample_bytree=cs, n_estimators=rounds_for(codes[j]), **cfg).fit(Xa[:, sel], ya, **fit_kw)
            if WANDB: cur.append((m.evals_result_["train"]["binary_logloss"], m.evals_result_["val"]["binary_logloss"]))
            if EXPORT and K == KS[0]: m.booster_.save_model(str(EXPORT / f"{codes[j]}_fold{fi}.txt")); np.save(EXPORT / f"{codes[j]}_fold{fi}_cols.npy", sel)
            oof[b, j] = m.predict_proba(X["train"][b][:, sel])[:, 1]; pv[:, j] += m.predict_proba(X["val"][:, sel])[:, 1] / 5; pt[:, j] += m.predict_proba(X["test"][:, sel])[:, 1] / 5
        if WANDB and cur and K == KS[0]:
            L = max(len(c[0]) for c in cur); pad = lambda v: np.r_[v, [np.nan] * (L - len(v))]
            tr_c = np.nanmean([pad(c[0]) for c in cur], 0); va_c = np.nanmean([pad(c[1]) for c in cur], 0)
            for r in range(L): wandb.log({"round": r + 1, f"curves/fold{fi}_train_logloss": float(tr_c[r]), f"curves/fold{fi}_val_logloss": float(va_c[r])})
    a_ = aps(oof); av = aps(pv, "val"); res[f"k{K}"] = a_
    if WANDB and K == KS[0]:
        wandb.summary.update({"oof_mAP": float(a_.mean()), "val_mAP": float(av.mean()), "oof_mAP_acc": float(a_[ai].mean()), "oof_mAP_term": float(a_[ti].mean()), "oof_mAP_chain": float(a_[ci].mean()), "val_mAP_acc": float(av[ai].mean()), "val_mAP_term": float(av[ti].mean()), "val_mAP_chain": float(av[ci].mean())})
        wandb.log({"per_target": wandb.Table(columns=["code", "oof_AP", "val_AP"], data=[[c, float(a_[j]), float(av[j])] for j, c in enumerate(codes)])})
    print(f"RESULT screen k={K:4d} (+{len(always)} always): OOF mAP={a_.mean():.4f} | acc {a_[ai].mean():.4f} / term {a_[ti].mean():.4f} / chain {a_[ci].mean():.4f} | val {av.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
    for s, arr in (("oof", oof), ("val", pv), ("test", pt)): np.save(OUT / f"k{K}{NSFX}_{s}.npy", arr)
ref = aps(np.load(HERE / "clean" / "oof.npy"))
per = pd.DataFrame({"cond": [desc[c][:34] for c in codes], "pos": Yv["train"].sum(0), **{k: v.round(3) for k, v in res.items()}, "lgb_clean": ref.round(3)}, index=codes)
pd.set_option("display.width", 250); print(per.sort_values("lgb_clean").to_string()); per.to_csv(OUT / "per_condition_screen.csv"); print(f"done [{time.time()-t0:.0f}s]")
if WANDB: wandb.finish()
