"""Upper-bound (oracle) and sample-size (learning-curve) decomposition of the OOF mAP on the official rows.
Question answered: of the gap between our 0.23 and 1.0, how much is (a) irreducible generator randomness,
(b) information that only exists inside the 5-year window (illegal for any leak-free model), (c) finite-sample
estimation error that a better model / more data could recover?
Configs (same LightGBM as lgb_clean, 5-fold patient OOF, at-risk masking):
  base     : pre-anchor clean features (2317 cols)
  odeath   : + oracle death-in-window / exposure            (legal models already get this at AUROC 0.998)
  owin     : + oracle in-window event tokens (COND excl. the 40 targets, MED, PROC, ENC class), first-in-window + counts
  owinsib  : + oracle first-in-window indicators of the other 39 targets
  lc25/50/75: base features, fold-training patients subsampled to 25/50/75 % (learning curve)
Outputs oracle/{cfg}_oof.npy, oracle/per_condition_oracle.csv, oracle/summary.json."""
import time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "oracle"; OUT.mkdir(exist_ok=True)
d = load(); pats, ev, Y, P, codes, targets = d["pats"], d["events"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
ptr = pats[pats.split == "train"].reset_index(drop=True); ids = ptr.PATIENT.values; n = len(ids)
X = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(ids).values.astype(np.float32)
Ytr, Ptr = Y.loc[ids].values, P.loc[ids].values.astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
grp = np.array(["term" if j in ti else "acc" if j in ai else "chain" for j in range(40)])
# ---- oracle death / exposure -------------------------------------------------------------------------------------
endd = ptr["end"].values.astype("datetime64[ns]"); anc = ptr.anchor.values.astype("datetime64[ns]"); dd = ptr.DEATHDATE.values.astype("datetime64[ns]")
diw = ptr.death_in_window.values.astype(float); died = ptr.died.values.astype(float)
t_death = (dd - anc) / np.timedelta64(1, "D") / 365.25; expo = np.where(np.isnan(t_death), 1.0, np.clip(t_death / 5.0, 0, 1))
gap_end = (endd - dd) / np.timedelta64(1, "D")
print(f"train n={n}; died {died.mean():.3f}; death in window {diw.mean():.3f}; among those, |end - deathdate| <= 1 day: {np.mean(np.abs(gap_end[diw == 1]) <= 1):.3f}, median exposure {np.median(expo[diw == 1]):.3f}", flush=True)
D = np.c_[diw, died, expo].astype(np.float32)
# ---- oracle in-window tokens ---------------------------------------------------------------------------------------
e = ev[ev.PATIENT.isin(ids)].merge(ptr[["PATIENT", "end"]], on="PATIENT")
e = e[e.kind.isin(["COND", "MED", "PROC", "ENC"])].copy()
e["tok"] = np.where(e.kind == "ENC", "ENC:" + e.encclass.astype(str), e.kind + ":" + e.code)
e = e[~(e.kind.eq("COND") & e.code.isin(codes))]
first = e.groupby(["PATIENT", "tok"]).date.min().reset_index().merge(ptr[["PATIENT", "anchor", "end"]], on="PATIENT")
fw = first[(first.date >= first.anchor) & (first.date <= first["end"])]
Wf = pd.crosstab(fw.PATIENT, fw.tok).reindex(ids).fillna(0).gt(0)
Wf = Wf.loc[:, Wf.sum() >= 20]
win = e[(e.date >= e.anchor) & (e.date <= e["end"])]
Wc = pd.crosstab(win.PATIENT, win.tok).reindex(ids).fillna(0); Wc = Wc.loc[:, (Wc > 0).sum() >= 20]
kcnt = pd.crosstab(win.PATIENT, win.kind).reindex(ids).fillna(0)
W = np.c_[Wf.values.astype(np.float32), np.log1p(Wc.values).astype(np.float32), np.log1p(kcnt.values).astype(np.float32)]
print(f"window oracle: {Wf.shape[1]} first-in-window tokens, {Wc.shape[1]} count tokens [{time.time()-t0:.0f}s]", flush=True)
# ---- runs ------------------------------------------------------------------------------------------------------------
cfg = dict(objective="binary", verbose=-1, n_jobs=12, learning_rate=0.03, n_estimators=350, subsample=0.8, subsample_freq=1, reg_lambda=5.0, num_leaves=8, min_child_samples=15, colsample_bytree=0.3, random_state=0)
folds = list(KFold(5, shuffle=True, random_state=0).split(ids))
def aps(p): p = np.where(Ptr, 0, p); return np.array([average_precision_score(Ytr[:, j], p[:, j]) if Ytr[:, j].sum() else 0 for j in range(40)])
def run(name, F, sib=False, frac=1.0):
    oof = np.zeros((n, 40))
    for fi, (a, b) in enumerate(folds):
        if frac < 1: a = np.random.RandomState(100 + fi).choice(a, int(round(frac * len(a))), replace=False)
        for j in range(40):
            m = ~Ptr[a, j]
            Fa = F
            if sib: oth = [k for k in range(40) if k != j]; Fa = np.c_[F, Ytr[:, oth]]
            mdl = lgb.LGBMClassifier(**cfg).fit(Fa[a][m], Ytr[a, j][m]); oof[b, j] = mdl.predict_proba(Fa[b])[:, 1]
    a_ = aps(oof); np.save(OUT / f"{name}_oof.npy", oof)
    print(f"RESULT {name:8s} feats {F.shape[1]+(39 if sib else 0):5d}: OOF mAP={a_.mean():.4f} | acc {a_[ai].mean():.4f} / term {a_[ti].mean():.4f} / chain {a_[ci].mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
    return a_
prev = np.array([Ytr[~Ptr[:, j], j].mean() for j in range(40)])
print(f"RESULT random  : AP=prevalence among at-risk: mAP={prev.mean():.4f} | acc {prev[ai].mean():.4f} / term {prev[ti].mean():.4f} / chain {prev[ci].mean():.4f}", flush=True)
res = {"prevalence": prev}
res["base"] = run("base", X)
res["odeath"] = run("odeath", np.c_[X, D])
res["owin"] = run("owin", np.c_[X, D, W])
res["owinsib"] = run("owinsib", np.c_[X, D, W], sib=True)
for f in (0.25, 0.5, 0.75): res[f"lc{int(f*100)}"] = run(f"lc{int(f*100)}", X, frac=f)
per = pd.DataFrame({"cond": [desc[c][:34] for c in codes], "grp": grp, "pos": Ytr.sum(0), **{k: v.round(3) for k, v in res.items()}}, index=codes)
pd.set_option("display.width", 250); print(per.sort_values(["grp", "owin"]).to_string()); per.to_csv(OUT / "per_condition_oracle.csv")
summ = {k: dict(all=float(v.mean()), acc=float(v[ai].mean()), term=float(v[ti].mean()), chain=float(v[ci].mean())) for k, v in res.items()}
json.dump(summ, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
