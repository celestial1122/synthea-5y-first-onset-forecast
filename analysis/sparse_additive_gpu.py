"""GPU version of the sparse-additive inductive-bias test on the clean dataset (5-fold x 2 OOF):
  (i) L1-penalized logistic regression for all 40 targets at once (PyTorch, full-batch Adam, at-risk masks), 3 penalties
  (ii) boosted stumps = XGBoost CUDA with max_depth=1, 1500 rounds
Outputs analysis/v4/{l1_lam*,stumps}_{oof,val,test}.npy for the assembler."""
import time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, torch, xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time(); dev = torch.device("cuda")
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; import genfeat; OUT = genfeat.outdir("v4"); OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
X = genfeat.augment(pd.read_parquet(HERE / "clean" / "X.parquet")).reindex(pats.PATIENT)
num_cols = [c for c in X.columns if c.split("_")[0] in ("last", "max", "min", "m5", "std", "trend", "rec", "first")]
Xl = pd.concat([X.fillna(0), X[num_cols].isna().astype(np.float32).add_prefix("miss_")], axis=1)
cnt_like = [c for c in Xl.columns if c.split("_")[0] in ("cnt", "c5", "c1", "nobs", "n")]; Xl[cnt_like] = np.log1p(Xl[cnt_like])
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Ytr, Ptr, Yva, Pva = Y.values[tr].astype(np.float32), P.values[tr].astype(bool), Y.values[va], P.values[va].astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
def aps(p, Yv, Pv):
    p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
folds = [(r, a, b) for r in range(2) for a, b in KFold(5, shuffle=True, random_state=r).split(np.arange(tr.sum()))]
XL = {"tr": Xl.values[tr].astype(np.float32), "va": Xl.values[va].astype(np.float32), "te": Xl.values[te].astype(np.float32)}

def l1_fit_predict(Xa, Ya, Ma, lam, Xs, steps=400, lr=0.02):
    mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
    f = lambda Z: torch.tensor(np.clip((Z - mu) / sd, -6, 6), device=dev)
    xa, ya, ma = f(Xa), torch.tensor(Ya, device=dev), torch.tensor(Ma.astype(np.float32), device=dev)
    prior = (ya * ma).sum(0).clamp_min(0.5) / ma.sum(0).clamp_min(1)
    W = torch.zeros(xa.shape[1], 40, device=dev, requires_grad=True); b = torch.log(prior / (1 - prior)).clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        logits = xa @ W + b
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, ya, reduction="none")
        loss = ((bce * ma).sum(0) / ma.sum(0).clamp_min(1)).sum() + lam * W.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():  # proximal soft-threshold keeps true sparsity
            W.copy_(torch.sign(W) * torch.clamp(W.abs() - lr * lam, min=0))
    with torch.no_grad():
        return [torch.sigmoid(f(Z) @ W + b).cpu().numpy() for Z in Xs], int((W.abs() > 1e-6).sum(0).float().mean().item())

res = {}
for lam in (3e-4, 1e-3, 3e-3):
    oof = np.zeros((tr.sum(), 40)); pv = np.zeros((va.sum(), 40)); pt = np.zeros((te.sum(), 40)); nz = []
    for r, a, b in folds:
        (po, pvv, ptt), k = l1_fit_predict(XL["tr"][a], Ytr[a], ~Ptr[a], lam, [XL["tr"][b], XL["va"], XL["te"]]); nz.append(k)
        oof[b] += po / 2; pv += pvv / len(folds); pt += ptt / len(folds)
    name = f"l1_lam{lam:g}"; a_ = aps(oof, Ytr, Ptr); v_ = aps(pv, Yva, Pva); res[name] = (a_, v_)
    print(f"{name:16s} OOF mAP={a_.mean():.4f} (acc {a_[ai].mean():.3f}/term {a_[ti].mean():.3f}/chain {a_[ci].mean():.3f}) | val {v_.mean():.4f} | nonzero coefs/target ~{int(np.mean(nz))}  [{time.time()-t0:.0f}s]", flush=True)
    np.save(OUT / f"{name}_oof.npy", oof); np.save(OUT / f"{name}_val.npy", pv); np.save(OUT / f"{name}_test.npy", pt)
# boosted stumps on GPU
Xg = {"tr": X.values[tr], "va": X.values[va], "te": X.values[te]}
oof = np.zeros((tr.sum(), 40)); pv = np.zeros((va.sum(), 40)); pt = np.zeros((te.sum(), 40))
for r, a, b in folds:
    for j in range(40):
        m = ~Ptr[a, j]; y = Ytr[a, j][m]
        mdl = xgb.XGBClassifier(device="cuda", tree_method="hist", max_depth=1, n_estimators=1500, learning_rate=0.02, subsample=0.7, colsample_bytree=0.3, min_child_weight=5, reg_lambda=5.0, verbosity=0).fit(Xg["tr"][a][m], y)
        oof[b, j] += mdl.predict_proba(Xg["tr"][b])[:, 1] / 2; pv[:, j] += mdl.predict_proba(Xg["va"])[:, 1] / len(folds); pt[:, j] += mdl.predict_proba(Xg["te"])[:, 1] / len(folds)
a_ = aps(oof, Ytr, Ptr); v_ = aps(pv, Yva, Pva); res["stumps"] = (a_, v_)
print(f"{'stumps (XGB d=1)':16s} OOF mAP={a_.mean():.4f} (acc {a_[ai].mean():.3f}/term {a_[ti].mean():.3f}/chain {a_[ci].mean():.3f}) | val {v_.mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
np.save(OUT / "stumps_oof.npy", oof); np.save(OUT / "stumps_val.npy", pv); np.save(OUT / "stumps_test.npy", pt)
g = np.load(HERE / "clean" / "oof.npy"); print(f"{'lgb clean (ref)':16s} OOF mAP={aps(g, Ytr, Ptr).mean():.4f}")
per = pd.DataFrame({"cond": [desc[c][:32] for c in codes], "lgb": aps(g, Ytr, Ptr), **{k: v[0] for k, v in res.items()}}, index=codes).round(3)
pd.set_option("display.width", 220); print(per.to_string()); per.to_csv(OUT / "per_condition_additive.csv")
json.dump({k: {"oof": float(v[0].mean()), "val": float(v[1].mean())} for k, v in res.items()}, open(OUT / "summary_additive.json", "w"), indent=1)
