"""Small GBM configuration grid judged by 5-fold OOF mAP on the 2791 training patients (not the 365 val),
plus duplicate-target averaging post-processing. Reports val as a secondary number."""
import warnings, time, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from feats import build_features
warnings.filterwarnings("ignore")
t0 = time.time()
d, X = build_features()
pats, codes = d["pats"], d["codes"]
tr = (pats.split == "train").values; va = (pats.split == "val").values
Y = d["Y"].values; P = d["P"].values.astype(bool)
Xtr, Ytr, Ptr = X.values[tr], Y[tr], P[tr]; Xva, Yva, Pva = X.values[va], Y[va], P[va]
print("X", X.shape, f"{time.time()-t0:.0f}s", flush=True)
GROUPS = [["162573006", "424132000", "254637007"], ["22325002", "723857007"], ["126906006", "92691004"]]
gidx = [[codes.index(c) for c in g] for g in GROUPS]

def mAP(probs, Yv, Pv, groups=False):
    probs = np.where(Pv, 0, probs.copy())
    if groups:
        for g in gidx:
            m = probs[:, g].mean(1, keepdims=True); probs[:, g] = np.where(Pv[:, g], 0, m)
    return np.array([average_precision_score(Yv[:, j], probs[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])

CONFIGS = {
    "base":      dict(learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=350),
    "shallow":   dict(learning_rate=0.02, num_leaves=4, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.2, reg_lambda=10.0, n_estimators=600),
    "stumps":    dict(learning_rate=0.03, num_leaves=2, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=600),
    "deeper":    dict(learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=300),
    "lowcol":    dict(learning_rate=0.02, num_leaves=6, min_child_samples=20, subsample=0.7, subsample_freq=1, colsample_bytree=0.1, reg_lambda=5.0, n_estimators=800),
}
results = {}
kf = list(KFold(5, shuffle=True, random_state=0).split(Xtr))
for name, cfg in CONFIGS.items():
    params = dict(objective="binary", verbose=-1, random_state=0, n_jobs=8, **cfg)
    oof = np.zeros_like(Ytr, dtype=float); pval = np.zeros_like(Yva, dtype=float)
    for a, b in kf:
        for j in range(40):
            m = ~Ptr[a, j]; y = Ytr[a, j][m]
            oof[b, j] = lgb.LGBMClassifier(**params).fit(Xtr[a][m], y).predict_proba(Xtr[b])[:, 1]
    for j in range(40):
        m = ~Ptr[:, j]
        pval[:, j] = lgb.LGBMClassifier(**params).fit(Xtr[m], Ytr[m, j]).predict_proba(Xva)[:, 1]
    r = dict(oof=mAP(oof, Ytr, Ptr).mean(), oof_grp=mAP(oof, Ytr, Ptr, True).mean(), val=mAP(pval, Yva, Pva).mean(), val_grp=mAP(pval, Yva, Pva, True).mean())
    results[name] = r
    np.save(f"gbm_{name}_oof.npy", oof); np.save(f"gbm_{name}_val.npy", pval)
    print(f"{name:10s} OOF mAP={r['oof']:.4f} (+group-avg {r['oof_grp']:.4f}) | val mAP={r['val']:.4f} (+group-avg {r['val_grp']:.4f})  [{time.time()-t0:.0f}s]", flush=True)
# ensemble of all configs
oofs = np.mean([np.load(f"gbm_{n}_oof.npy") for n in CONFIGS], 0); vals = np.mean([np.load(f"gbm_{n}_val.npy") for n in CONFIGS], 0)
print(f"{'cfg-ensemble':10s} OOF mAP={mAP(oofs, Ytr, Ptr).mean():.4f} (+group-avg {mAP(oofs, Ytr, Ptr, True).mean():.4f}) | val mAP={mAP(vals, Yva, Pva).mean():.4f} (+group-avg {mAP(vals, Yva, Pva, True).mean():.4f})")
json.dump(results, open("tune_gbm_results.json", "w"), indent=1)
