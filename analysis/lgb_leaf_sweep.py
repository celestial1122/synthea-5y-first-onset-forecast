"""LightGBM leaf-constraint sweep on the clean dataset (the XGBoost runs showed rare targets are very sensitive to the
minimum-leaf constraint). 6 configs x 5-fold x 2 reps; rep0 selects per target, rep1 evaluates."""
import time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "lgb_sweep"; OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
X = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(pats.PATIENT)
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Xtr, Xva, Xte = X.values[tr], X.values[va], X.values[te]; Ytr, Ptr, Yva, Pva = Y.values[tr], P.values[tr].astype(bool), Y.values[va], P.values[va].astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
def aps(p, Yv, Pv):
    p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
folds = {r: list(KFold(5, shuffle=True, random_state=r).split(Xtr)) for r in range(2)}
COMMON = dict(objective="binary", verbose=-1, n_jobs=12, learning_rate=0.03, n_estimators=350, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0)
GRID = {f"nl{nl}_mcs{mcs}": dict(COMMON, num_leaves=nl, min_child_samples=mcs) for nl in (4, 8) for mcs in (5, 15, 40)}
res = {}
for name, cfg in GRID.items():
    oofs = [np.zeros((len(Xtr), 40)) for _ in range(2)]; pv = np.zeros((len(Xva), 40)); pt = np.zeros((len(Xte), 40))
    for r in range(2):
        for a, b in folds[r]:
            for j in range(40):
                m = ~Ptr[a, j]; mdl = lgb.LGBMClassifier(**cfg).fit(Xtr[a][m], Ytr[a, j][m])
                oofs[r][b, j] = mdl.predict_proba(Xtr[b])[:, 1]; pv[:, j] += mdl.predict_proba(Xva)[:, 1] / 10; pt[:, j] += mdl.predict_proba(Xte)[:, 1] / 10
    res[name] = (oofs, pv, pt); a0, a1 = aps(oofs[0], Ytr, Ptr), aps(oofs[1], Ytr, Ptr)
    print(f"{name:12s} OOF rep0={a0.mean():.4f} rep1={a1.mean():.4f} (acc {a1[ai].mean():.3f}/term {a1[ti].mean():.3f}/chain {a1[ci].mean():.3f}) | val {aps(pv, Yva, Pva).mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    np.save(OUT / f"{name}_oof.npy", np.mean(oofs, 0)); np.save(OUT / f"{name}_val.npy", pv); np.save(OUT / f"{name}_test.npy", pt)
names = list(GRID); avg = [np.mean([res[n][0][r] for n in names], 0) for r in range(2)]
print(f"{'all-config avg':12s} OOF rep0={aps(avg[0], Ytr, Ptr).mean():.4f} rep1={aps(avg[1], Ytr, Ptr).mean():.4f} | val {aps(np.mean([res[n][1] for n in names], 0), Yva, Pva).mean():.4f}")
np.save(OUT / "avg_oof.npy", np.mean(avg, 0)); np.save(OUT / "avg_val.npy", np.mean([res[n][1] for n in names], 0)); np.save(OUT / "avg_test.npy", np.mean([res[n][2] for n in names], 0))
pick = []; sel1 = np.zeros((len(Xtr), 40))
for j in range(40):
    s0 = {n: average_precision_score(Ytr[:, j], np.where(Ptr[:, j], 0, res[n][0][0][:, j])) for n in names}; n = max(s0, key=s0.get); pick.append(n); sel1[:, j] = res[n][0][1][:, j]
print(f"per-target pick (rep0) -> honest rep1 OOF={aps(sel1, Ytr, Ptr).mean():.4f}; picks: {pd.Series(pick).value_counts().to_dict()}")
per = pd.DataFrame({"cond": [desc[c][:32] for c in codes], "train_pos": [int(Ytr[~Ptr[:, j], j].sum()) for j in range(40)], **{n: np.mean([aps(res[n][0][r], Ytr, Ptr) for r in range(2)], 0) for n in names}}, index=codes).round(3)
pd.set_option("display.width", 220); print(per.to_string()); per.to_csv(OUT / "per_condition_lgb_sweep.csv")
json.dump({n: float(aps(res[n][0][1], Ytr, Ptr).mean()) for n in names}, open(OUT / "summary.json", "w"), indent=1); print("done", f"{time.time()-t0:.0f}s")
