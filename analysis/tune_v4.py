"""Last legitimate model-side levers on the clean dataset, judged on train OOF (rep0 for any selection, rep1 for honest evaluation):
  (a) GBM 2-config baseline
  (b) fold-internal univariate screening (top-150 features by |AUROC-0.5| on the fold's training part) -> GBM
  (c) 5-seed bagged GBM (subsample 0.7, colsample 0.2)
  (d) ExtraTrees (different model family) and the average of all
  (e) per-target choice among {a, b, c, d, avg, age/sex LR} selected on rep0 OOF, evaluated on rep1 OOF
"""
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "v4"; OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
X = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(pats.PATIENT)
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Xtr, Xva, Xte = X.values[tr], X.values[va], X.values[te]; Ytr, Ptr = Y.values[tr], P.values[tr].astype(bool); Yva, Pva = Y.values[va], P.values[va].astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
def aps(p, Yv, Pv):
    p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
def rep(name, oof_by_rep, pv):
    a = [aps(o, Ytr, Ptr) for o in oof_by_rep]; m = np.mean(a, 0); v = aps(pv, Yva, Pva)
    print(f"{name:44s} OOF mAP rep0={a[0].mean():.4f} rep1={a[1].mean():.4f} mean={m.mean():.4f} (acc {m[ai].mean():.3f}/term {m[ti].mean():.3f}/chain {m[ci].mean():.3f}) | val {v.mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return a, v
NREP, NFOLD = 2, 5
folds = {rep_: list(KFold(NFOLD, shuffle=True, random_state=rep_).split(Xtr)) for rep_ in range(NREP)}
import xgboost as xgb
BASE = dict(device="cuda", tree_method="hist", n_estimators=350, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.3, reg_lambda=5.0, verbosity=0)
SHAL = dict(BASE, learning_rate=0.02, max_depth=2, min_child_weight=10, colsample_bytree=0.2, reg_lambda=10.0, n_estimators=600)

def cv(make, screen=None):
    """returns oof per rep (list of n_tr x 40), val/test averaged over all fold models"""
    oofs = [np.zeros((len(Xtr), 40)) for _ in range(NREP)]; pv = np.zeros((len(Xva), 40)); pt = np.zeros((len(Xte), 40)); n = 0
    for r in range(NREP):
        for a, b in folds[r]:
            for j in range(40):
                m = ~Ptr[a, j]; y = Ytr[a, j][m]; Xa = Xtr[a][m]
                cols = slice(None)
                if screen is not None:
                    xs = np.nan_to_num(Xa); sc = np.array([abs(roc_auc_score(y, xs[:, k]) - 0.5) if xs[:, k].std() > 0 else 0 for k in range(xs.shape[1])])
                    cols = np.argsort(-sc)[:screen]
                mdl = make().fit(Xa[:, cols], y)
                f = lambda Z: Z[:, cols]
                oofs[r][b, j] = mdl.predict_proba(f(Xtr[b]))[:, 1]; pv[:, j] += mdl.predict_proba(f(Xva))[:, 1]; pt[:, j] += mdl.predict_proba(f(Xte))[:, 1]
            n += 1
    return oofs, pv / n, pt / n

res = {}
res["a_gbm2"] = cv(lambda: xgb.XGBClassifier(**BASE)); oofs2, pv2, pt2 = cv(lambda: xgb.XGBClassifier(**SHAL))
res["a_gbm2"] = ([(x + y) / 2 for x, y in zip(res["a_gbm2"][0], oofs2)], (res["a_gbm2"][1] + pv2) / 2, (res["a_gbm2"][2] + pt2) / 2); rep("(a) GBM 2-config", *res["a_gbm2"][:2])
res["b_screen"] = cv(lambda: xgb.XGBClassifier(**BASE), screen=150); rep("(b) screened top-150 -> GBM", *res["b_screen"][:2])
bag = [cv(lambda s=s: xgb.XGBClassifier(**dict(BASE, subsample=0.7, colsample_bytree=0.2, random_state=s))) for s in range(5)]
res["c_bag5"] = ([np.mean([b_[0][r] for b_ in bag], 0) for r in range(NREP)], np.mean([b_[1] for b_ in bag], 0), np.mean([b_[2] for b_ in bag], 0)); rep("(c) 5-seed bagged GBM", *res["c_bag5"][:2])
res["d_et"] = cv(lambda: xgb.XGBRFClassifier(n_estimators=300, max_depth=6, subsample=0.8, colsample_bynode=0.1, learning_rate=1.0, min_child_weight=5, device="cuda", tree_method="hist", verbosity=0, random_state=0)); rep("(d) XGB random forest (GPU)", *res["d_et"][:2])
keys = ["a_gbm2", "b_screen", "c_bag5", "d_et"]
res["avg"] = ([np.mean([res[k][0][r] for k in keys], 0) for r in range(NREP)], np.mean([res[k][1] for k in keys], 0), np.mean([res[k][2] for k in keys], 0)); rep("average of (a)-(d)", *res["avg"][:2])
# age/sex LR
def S(mask): a = pats.age.values[mask]; return np.c_[a, a ** 2, pats.female.values[mask]]
sc = StandardScaler().fit(S(tr)); Str, Sva, Ste = sc.transform(S(tr)), sc.transform(S(va)), sc.transform(S(te))
lr_oofs = [np.zeros((len(Xtr), 40)) for _ in range(NREP)]; lr_v = np.zeros((len(Xva), 40)); lr_t = np.zeros((len(Xte), 40)); n = 0
for r in range(NREP):
    for a, b in folds[r]:
        for j in range(40):
            m = ~Ptr[a, j]; mdl = LogisticRegression(C=1.0, max_iter=2000).fit(Str[a][m], Ytr[a, j][m]); lr_oofs[r][b, j] = mdl.predict_proba(Str[b])[:, 1]; lr_v[:, j] += mdl.predict_proba(Sva)[:, 1]; lr_t[:, j] += mdl.predict_proba(Ste)[:, 1]
        n += 1
res["lr"] = (lr_oofs, lr_v / n, lr_t / n); rep("age/sex LR", *res["lr"][:2])
# (e) per-target selection on rep0, honest evaluation on rep1
cands = ["a_gbm2", "b_screen", "c_bag5", "d_et", "avg", "lr"]
choice = []; sel_oof1 = np.zeros((len(Xtr), 40)); sel_v = np.zeros((len(Xva), 40)); sel_t = np.zeros((len(Xte), 40))
for j in range(40):
    s0 = {k: average_precision_score(Ytr[:, j], np.where(Ptr[:, j], 0, res[k][0][0][:, j])) for k in cands}; k = max(s0, key=s0.get); choice.append(k)
    sel_oof1[:, j] = res[k][0][1][:, j]; sel_v[:, j] = res[k][1][:, j]; sel_t[:, j] = res[k][2][:, j]
a1 = aps(sel_oof1, Ytr, Ptr); print(f"(e) per-target choice (selected on rep0) -> honest rep1 OOF mAP={a1.mean():.4f} | val {aps(sel_v, Yva, Pva).mean():.4f}; choices: {pd.Series(choice).value_counts().to_dict()}")
print("   vs avg rep1:", round(aps(res['avg'][0][1], Ytr, Ptr).mean(), 4), "| vs GBM2 rep1:", round(aps(res['a_gbm2'][0][1], Ytr, Ptr).mean(), 4))
per = pd.DataFrame({"cond": [desc[c][:32] for c in codes], **{k: np.mean([aps(o, Ytr, Ptr) for o in res[k][0]], 0) for k in cands}, "choice": choice}, index=codes).round(3)
pd.set_option("display.width", 220); print(per.to_string()); per.to_csv(OUT / "per_condition_v4.csv")
for k in ("avg", "c_bag5"):
    np.save(OUT / f"{k}_oof.npy", np.mean(res[k][0], 0)); np.save(OUT / f"{k}_val.npy", res[k][1]); np.save(OUT / f"{k}_test.npy", res[k][2])
json.dump({k: {"oof_mean": float(np.mean([aps(o, Ytr, Ptr).mean() for o in res[k][0]])), "val": float(aps(res[k][1], Yva, Pva).mean())} for k in cands}, open(OUT / "summary_v4.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
