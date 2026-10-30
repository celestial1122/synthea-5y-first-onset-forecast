"""Death-gated mixture of experts on the person-period dataset (GPU XGBoost, same folds/outputs as pp_sweep2.py).
    P(y_j | x) = g(x) * E_D(x) + (1 - g(x)) * E_A(x)
E_D : expert fitted on official rows of training patients who die inside the window (the 'last 5 years of life' regime)
E_A : expert fitted on every other in-fold row (official rows of survivors + all landmark rows, where death is impossible)
g   : cross-fitted P(death in window | pre-anchor history) from stack/pdeath_*.npy (OOF AUROC 0.998), learned once from
      1,199 deaths instead of being rediscovered by every 50-230-positive target.
Usage: pp_mix.py [--cfg d3] [--ds ppds] [--mode mix|pdfeat]   (pdfeat = logit P(death) as a feature, landmark rows = logit(1e-3))
Outputs ppds/mix_<cfg>_{oof,val,test}.npy (official-row layout = pats order within split, same as other candidates)."""
import time, json, argparse, warnings, gc
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore"); t0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", default="d3"); ap.add_argument("--ds", default="ppds"); ap.add_argument("--mode", default="mix"); ap.add_argument("--tag", default="")
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DS = HERE / args.ds; DATA = ROOT / "research_intern_coding_test"
CFGS = {"d3": dict(max_depth=3, n=400, eta=0.03, colsample_bytree=0.3, min_child_weight=5), "d6": dict(max_depth=6, n=600, eta=0.03, colsample_bytree=0.3, min_child_weight=5)}
X = pd.read_parquet(DS / "X.parquet").set_index("PATIENT"); M = np.load(DS / "meta.npz", allow_pickle=True)
y, prior, k, split, pid, codes = M["y"], M["prior"].astype(bool), M["k"], M["split"], M["pid"].astype(str), list(M["codes"])
# ---- death-in-window flag and cross-fitted gate, aligned to ppds rows -------------------------------------------------
pa = pd.concat([pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id", "DEATHDATE"]), pd.read_csv(DATA / "test/patients.csv", dtype=str, usecols=["Id", "DEATHDATE"])], ignore_index=True).rename(columns={"Id": "PATIENT"})
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"}); pa = pa.merge(spl, on="PATIENT", how="inner")   # pats order of ehr_load.load()
died = pa.set_index("PATIENT").DEATHDATE.notna()
gate = {}
for s in ("train", "val", "test"):
    ids = pa[pa.split == s].PATIENT.values; arr = np.load(HERE / "stack" / f"pdeath_{'oof' if s == 'train' else s}.npy"); assert len(arr) == len(ids)
    gate[s] = pd.Series(arr, index=ids)
off = {s: (k == 0) & (split == s) for s in ("train", "val", "test")}
diw_tr = died.reindex(pid[off["train"]]).values.astype(float)                       # for train patients died == death in window (last encounter = death certification)
g_row = np.zeros(len(pid));
for s in ("train", "val", "test"): g_row[off[s]] = gate[s].reindex(pid[off[s]]).values
print(f"mode {args.mode}: rows {len(pid)}, official train {off['train'].sum()} (dying {int(diw_tr.sum())}), gate aligned (train gate>0.5 vs died agreement {np.mean((g_row[off['train']] > 0.5) == (diw_tr == 1)):.4f}) [{time.time()-t0:.0f}s]", flush=True)
if args.mode == "pdfeat":
    lg = np.full(len(pid), np.log(1e-3 / (1 - 1e-3))); m_ = (k == 0); gg = np.clip(g_row[m_], 1e-4, 1 - 1e-4); lg[m_] = np.log(gg / (1 - gg)); X["logit_pdeath"] = lg.astype(np.float32)
Xv = np.ascontiguousarray(X.values, dtype=np.float32); del X; gc.collect()
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
train_ids = pid[off["train"]]; Ytr, Ptr = y[off["train"]], prior[off["train"]]; Yva, Pva = y[off["val"]], prior[off["val"]]
dva = xgb.DMatrix(Xv[off["val"]]); dte = xgb.DMatrix(Xv[off["test"]])
p = dict(CFGS[args.cfg]); NR = p.pop("n"); params = dict(objective="binary:logistic", tree_method="hist", device="cuda", subsample=0.8, reg_lambda=5.0, verbosity=0, seed=0, **p)
is_dying_off = np.zeros(len(pid), bool); is_dying_off[np.where(off["train"])[0][diw_tr == 1]] = True
oof = np.zeros((off["train"].sum(), 40)); pv = np.zeros((off["val"].sum(), 40)); pt = np.zeros((off["test"].sum(), 40))
for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(train_ids)):
    held = set(train_ids[b]); in_fold = ~np.isin(pid, list(held)) & ~((k == 0) & (split != "train"))
    rows_b = np.where(off["train"])[0][b]; db = xgb.DMatrix(Xv[rows_b]); gb = g_row[rows_b]; gv = g_row[off["val"]]; gt = g_row[off["test"]]
    if args.mode == "pdfeat":
        selA = in_fold; selD = None
    else:
        selA = in_fold & ~is_dying_off; selD = in_fold & is_dying_off
    dA = xgb.QuantileDMatrix(Xv[selA], max_bin=256); yA, pA = y[selA], prior[selA]
    if selD is not None: dD = xgb.QuantileDMatrix(Xv[selD], max_bin=256); yD, pD = y[selD], prior[selD]
    print(f"  fold {fold}: expert-A rows {int(selA.sum())}" + (f", expert-D rows {int(selD.sum())}" if selD is not None else "") + f" [{time.time()-t0:.0f}s]", flush=True)
    for j in range(40):
        at = ~pA[:, j]; yy = yA[:, j].astype(np.float32)
        if yy[at].sum() < 5: continue
        dA.set_label(yy); dA.set_weight(at.astype(np.float32)); bA = xgb.train(params, dA, num_boost_round=NR)
        eA = (bA.predict(db), bA.predict(dva), bA.predict(dte))
        if selD is None:
            pb, pva, pte = eA
        else:
            atD = ~pD[:, j]; yyD = yD[:, j].astype(np.float32)
            if yyD[atD].sum() < 5:
                c = max(float(yyD[atD].mean()) if atD.sum() else 0.0, 1e-4); eD = (np.full(len(rows_b), c), np.full(int(off["val"].sum()), c), np.full(int(off["test"].sum()), c))
            else:
                dD.set_label(yyD); dD.set_weight(atD.astype(np.float32)); bD = xgb.train(params, dD, num_boost_round=NR); eD = (bD.predict(db), bD.predict(dva), bD.predict(dte))
            pb = gb * eD[0] + (1 - gb) * eA[0]; pva = gv * eD[1] + (1 - gv) * eA[1]; pte = gt * eD[2] + (1 - gt) * eA[2]
        oof[b, j] = pb; pv[:, j] += pva / 5; pt[:, j] += pte / 5
    print(f"  fold {fold} done [{time.time()-t0:.0f}s]", flush=True); del dA; gc.collect()
def aps(p_, Yv_, Pv_): p_ = np.where(Pv_, 0, p_); return np.array([average_precision_score(Yv_[:, j], p_[:, j]) if Yv_[:, j].sum() else 0 for j in range(40)])
a_ = aps(oof, Ytr, Ptr); av = aps(pv, Yva, Pva); run = f"{args.mode}_{args.cfg}_all{args.tag}"
print(f"RESULT {run}: OOF mAP={a_.mean():.4f} | acc {a_[ai].mean():.4f} / term {a_[ti].mean():.4f} / chain {a_[ci].mean():.4f} | val {av.mean():.4f} [{time.time()-t0:.0f}s]")
np.save(DS / f"{run}_oof.npy", oof); np.save(DS / f"{run}_val.npy", pv); np.save(DS / f"{run}_test.npy", pt)
json.dump({"run": run, "oof_by_target": dict(zip(codes, a_.round(4).tolist())), "val_by_target": dict(zip(codes, av.round(4).tolist()))}, open(DS / f"{run}_summary.json", "w"), indent=1)
