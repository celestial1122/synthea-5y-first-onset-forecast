"""Death-gated mixture of experts on the official rows (law of total probability with a near-known gate):
    P(y_j | x) = P(D=1 | x) * P(y_j | D=1, x) + P(D=0 | x) * P(y_j | D=0, x)
D = dies inside the 5-year window (43 % of train; for dying patients the window is exactly the last 5 years of life, so
the 10 'terminal' targets are mostly cause-of-death attribution). The gate P(D | x) is learned ONCE from 1,199 deaths
(cross-fitted, stack/pdeath_*.npy, OOF AUROC 0.998); each expert is a LightGBM fitted on the dying / surviving training
patients only, so a 60-150-positive target no longer has to rediscover the death discriminator from its own positives.
Configs (5-fold patient OOF, same LightGBM as lgb_clean):  base | diw_feat (oracle 0/1 death as a feature, upper bound)
| pdeath_feat (cross-fitted P(death) as a feature) | mix (experts by true D, gate = cross-fitted P(death)) | mix_oracle (gate = true D).
Outputs mix/{cfg}_{oof,val,test}.npy (ensemble-candidate layout: pats order within split) and mix/per_condition_mix.csv."""
import time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "mix"; OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
Xall = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(pats.PATIENT)
S = {s: (pats.split == s).values for s in ("train", "val", "test")}
X = {s: Xall.values[S[s]].astype(np.float32) for s in S}; Yv = {s: Y.values[S[s]] for s in S}; Pv = {s: P.values[S[s]].astype(bool) for s in S}
diw = pats.death_in_window.values.astype(float)[S["train"]]
g = {"train": np.load(HERE / "stack/pdeath_oof.npy"), "val": np.load(HERE / "stack/pdeath_val.npy"), "test": np.load(HERE / "stack/pdeath_test.npy")}
print(f"train {S['train'].sum()} (dying {int(diw.sum())}), val {S['val'].sum()}, test {S['test'].sum()}; gate OOF AUROC {roc_auc_score(diw, g['train']):.4f}, "
      f"gate>0.5 accuracy {np.mean((g['train'] > 0.5) == (diw == 1)):.4f} [{time.time()-t0:.0f}s]", flush=True)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
cfg = dict(objective="binary", verbose=-1, n_jobs=12, learning_rate=0.03, n_estimators=350, subsample=0.8, subsample_freq=1, reg_lambda=5.0, num_leaves=8, min_child_samples=15, colsample_bytree=0.3, random_state=0)
folds = list(KFold(5, shuffle=True, random_state=0).split(X["train"]))
def aps(p, s="train"): p = np.where(Pv[s], 0, p); return np.array([average_precision_score(Yv[s][:, j], p[:, j]) if Yv[s][:, j].sum() else 0 for j in range(40)])
def fit_predict(Xa, ya, Xbs):
    """fit one expert; with < 5 positives fall back to the training prevalence"""
    if ya.sum() < 5: return [np.full(len(Xb), max(ya.mean(), 1e-4)) for Xb in Xbs]
    m = lgb.LGBMClassifier(**cfg).fit(Xa, ya); return [m.predict_proba(Xb)[:, 1] for Xb in Xbs]
def run(name, extra=None, mixture=False, oracle_gate=False):
    F = {s: (X[s] if extra is None else np.c_[X[s], extra[s]]) for s in S}
    oof = np.zeros((len(F["train"]), 40)); pv = np.zeros((len(F["val"]), 40)); pt = np.zeros((len(F["test"]), 40))
    for a, b in folds:
        for j in range(40):
            at = ~Pv["train"][a, j]; ya = Yv["train"][a, j]
            if not mixture:
                pb, pva, pte = fit_predict(F["train"][a][at], ya[at], [F["train"][b], F["val"], F["test"]])
            else:
                dead = diw[a] == 1
                eD = fit_predict(F["train"][a][at & dead], ya[at & dead], [F["train"][b], F["val"], F["test"]])
                eA = fit_predict(F["train"][a][at & ~dead], ya[at & ~dead], [F["train"][b], F["val"], F["test"]])
                gb = diw[b] if oracle_gate else g["train"][b]; gv, gt = g["val"], g["test"]
                pb = gb * eD[0] + (1 - gb) * eA[0]; pva = gv * eD[1] + (1 - gv) * eA[1]; pte = gt * eD[2] + (1 - gt) * eA[2]
            oof[b, j] = pb; pv[:, j] += pva / 5; pt[:, j] += pte / 5
    a_ = aps(oof); av = aps(pv, "val")
    print(f"RESULT {name:12s}: OOF mAP={a_.mean():.4f} | acc {a_[ai].mean():.4f} / term {a_[ti].mean():.4f} / chain {a_[ci].mean():.4f} | val {av.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
    for s, arr in (("oof", oof), ("val", pv), ("test", pt)): np.save(OUT / f"{name}_{s}.npy", arr)
    return a_
res = {}
res["base"] = run("base")
res["diw_feat"] = run("diw_feat", extra={"train": diw[:, None], "val": g["val"][:, None], "test": g["test"][:, None]})
lg = {s: np.log(np.clip(g[s], 1e-4, 1 - 1e-4) / (1 - np.clip(g[s], 1e-4, 1 - 1e-4)))[:, None] for s in S}
res["pdeath_feat"] = run("pdeath_feat", extra=lg)
res["mix"] = run("mix", mixture=True)
res["mix_oracle"] = run("mix_oracle", mixture=True, oracle_gate=True)
grp = np.array(["term" if j in ti else "acc" if j in ai else "chain" for j in range(40)])
per = pd.DataFrame({"cond": [desc[c][:34] for c in codes], "grp": grp, "pos": Yv["train"].sum(0), "pos_dying": (Yv["train"] * (diw[:, None] == 1)).sum(0), **{k: v.round(3) for k, v in res.items()}}, index=codes)
pd.set_option("display.width", 250); print(per.sort_values(["grp", "base"]).to_string()); per.to_csv(OUT / "per_condition_mix.csv")
json.dump({k: dict(all=float(v.mean()), acc=float(v[ai].mean()), term=float(v[ti].mean()), chain=float(v[ci].mean())) for k, v in res.items()}, open(OUT / "summary.json", "w"), indent=1)
print(f"done [{time.time()-t0:.0f}s]")
