"""Level-2 stacking (Wolpert 1992) on the level-1 OOF predictions: for every target, a small regularised model on
  [logit of own-target OOF from each family] + [cross-fitted P(death in window)] + [best family's OOF for the other 39 targets] + age/sex.
Rationale: the 40 targets are coupled through the death/era factor and competing causes; the per-target greedy ensemble
(assemble_final.py) cannot use cross-target information. Honest evaluation: level-2 is itself 5-fold cross-fitted (x3 seeds)
on the level-1 OOF rows; val/test use level-2 fitted on all train rows applied to level-1 val/test predictions."""
import json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; import genfeat; OUT = genfeat.outdir("stack"); OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
ids = {s: pats[pats.split == s].PATIENT.values for s in ("train", "val", "test")}
Yv = {s: Y.loc[ids[s]].values for s in ("train", "val")}; Pv = {s: P.loc[ids[s]].values.astype(bool) for s in ("train", "val", "test")}
died = pats.set_index("PATIENT").died.reindex(ids["train"]).values.astype(int)
X = genfeat.augment(pd.read_parquet(HERE / "clean" / "X.parquet")); Xs = {s: X.reindex(ids[s]) for s in ids}
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
def npy(dirn, stem):
    f = {s: HERE / dirn / f"{stem}{k}.npy" for s, k in (("train", "oof"), ("val", "val"), ("test", "test"))}
    return {s: np.load(p) for s, p in f.items()} if all(p.exists() for p in f.values()) else None
def csv(stem):
    f = {s: HERE / "cv_tf" / f"{stem}_{k}.csv" for s, k in (("train", "oof"), ("val", "val"), ("test", "test"))}
    return {s: pd.read_csv(p, dtype={"patient_id": str}).set_index("patient_id").reindex(ids[s])[codes].values for s, p in f.items()} if all(p.exists() for p in f.values()) else None
fam = {"lgb_clean": npy("clean", ""), "lgb_sweep_avg": npy("lgb_sweep", "avg_"), "xgb_v5avg": npy("v5", "avg_"), "pp_yearly": npy("pp2", "pp_yearly_uniform_"), "pp_lgb": npy("pp_lgb", "person-period_uniform_"),
       "pp_x5": npy("pp", "person-period_official_x5_"), "l1_lam0.003": npy("v4", "l1_lam0.003_"), "stumps": npy("v4", "stumps_"), "tf_hybrid": csv("hybrid")}
fam = {k: v for k, v in fam.items() if v is not None}; names = list(fam); print("families:", names)
nonterm_only = {"pp_yearly", "pp_lgb", "pp_x5"}
# cross-fitted death model (LightGBM on clean features) -> P(D | H) as a stacking feature
cfg = dict(objective="binary", verbose=-1, n_jobs=8, learning_rate=0.03, n_estimators=300, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0)
pD = {"train": np.zeros(len(ids["train"])), "val": np.zeros(len(ids["val"])), "test": np.zeros(len(ids["test"]))}
for a, b in KFold(5, shuffle=True, random_state=11).split(Xs["train"]):
    m = lgb.LGBMClassifier(**cfg).fit(Xs["train"].values[a], died[a]); pD["train"][b] = m.predict_proba(Xs["train"].values[b])[:, 1]
    pD["val"] += m.predict_proba(Xs["val"].values)[:, 1] / 5; pD["test"] += m.predict_proba(Xs["test"].values)[:, 1] / 5
print(f"death model OOF AUROC {roc_auc_score(died, pD['train']):.4f} [{time.time()-t0:.0f}s]")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def design(s, j):
    own = [lg(fam[k][s][:, j]) for k in names if not (k in nonterm_only and codes[j] in TERMINAL)]
    others = lg(np.delete(fam["lgb_sweep_avg"][s], j, axis=1))              # cross-target information from one family
    base = np.column_stack(own + [lg(pD[s]), Xs[s]["age"].values / 50.0, Xs[s]["female"].values])
    return base, np.column_stack([base, others])
def ap(y, p): return average_precision_score(y, p) if y.sum() else 0.0
res = {}
for variant in ("own+death", "own+death+others"):
    for model in ("lr", "gbm"):
        oof = np.zeros((len(ids["train"]), 40)); pv = np.zeros((len(ids["val"]), 40)); pt = np.zeros((len(ids["test"]), 40))
        for j in range(40):
            A = {s: design(s, j)[0 if variant == "own+death" else 1] for s in ("train", "val", "test")}
            y = Yv["train"][:, j]; atr = ~Pv["train"][:, j]
            def fit(Xa, ya):
                if model == "lr": return LogisticRegression(C=0.3, max_iter=2000).fit(Xa, ya)
                return lgb.LGBMClassifier(objective="binary", verbose=-1, n_jobs=8, learning_rate=0.03, n_estimators=150, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0).fit(Xa, ya)
            for r in range(3):
                for a, b in KFold(5, shuffle=True, random_state=100 + r).split(A["train"]):
                    aa = a[atr[a]]; mdl = fit(A["train"][aa], y[aa]); oof[b, j] += mdl.predict_proba(A["train"][b])[:, 1] / 3
            mdl = fit(A["train"][atr], y[atr]); pv[:, j] = mdl.predict_proba(A["val"])[:, 1]; pt[:, j] = mdl.predict_proba(A["test"])[:, 1]
        a = np.array([ap(Yv["train"][:, j], np.where(Pv["train"][:, j], 0, oof[:, j])) for j in range(40)]); av = np.array([ap(Yv["val"][:, j], np.where(Pv["val"][:, j], 0, pv[:, j])) for j in range(40)])
        name = f"{variant}_{model}"; res[name] = a
        print(f"L2 {name:24s} OOF mAP={a.mean():.4f} (acc {a[ai].mean():.3f} / term {a[ti].mean():.3f} / chain {a[ci].mean():.3f}) | val mAP={av.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
        np.save(OUT / f"{name}_oof.npy", oof); np.save(OUT / f"{name}_val.npy", pv); np.save(OUT / f"{name}_test.npy", pt)
# reference: best single family per target (oracle, optimistic) and greedy ensemble number from final/summary.json
best_single = np.array([max(ap(Yv["train"][:, j], np.where(Pv["train"][:, j], 0, fam[k]["train"][:, j])) for k in names if not (k in nonterm_only and codes[j] in TERMINAL)) for j in range(40)])
print(f"reference: per-target best single family (oracle) OOF={best_single.mean():.4f}; greedy ensemble cross-fitted OOF={json.load(open(HERE/'final'/'summary.json'))['oof_crossfit_mAP']:.4f}")
pd.DataFrame({"cond": [desc[c][:32] for c in codes], **{k: v.round(3) for k, v in res.items()}, "best_single": best_single.round(3)}, index=codes).to_csv(OUT / "per_condition_stack.csv")
np.save(OUT / "pdeath_oof.npy", pD["train"]); np.save(OUT / "pdeath_val.npy", pD["val"]); np.save(OUT / "pdeath_test.npy", pD["test"])
