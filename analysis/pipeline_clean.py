"""OOF evaluation on the cleaned dataset: GBM (2 configs, fold-ensembled), per-group report, sensitivity variants
(with/without QoL observations; with the leaked 'has DNA' bit, for the record only). Writes predictions_clean.csv."""
import json, sys, time, warnings, glob, os
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from ehr_load import load, DATA
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Yn, Pn = Y.values, P.values.astype(bool)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
GROUPS = [["162573006", "424132000", "254637007"], ["22325002", "723857007"]]
NREP, NFOLD = 2, 5
CFGS = {"base": dict(learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=350),
        "shallow": dict(learning_rate=0.02, num_leaves=4, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.2, reg_lambda=10.0, n_estimators=600)}

def aps(p, Yv, Pv):
    p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
def auc(p, Yv, Pv):
    p = np.where(Pv, 0, p); return float(np.mean([roc_auc_score(Yv[:, j], p[:, j]) for j in range(40)]))

def run(X, name):
    Xtr, Xva, Xte = X.values[tr], X.values[va], X.values[te]; Ytr, Ptr = Yn[tr], Pn[tr]
    folds = [(rep, a, b) for rep in range(NREP) for a, b in KFold(NFOLD, shuffle=True, random_state=rep).split(Xtr)]
    oof = np.zeros((len(Xtr), 40)); pv = np.zeros((len(Xva), 40)); pt = np.zeros((len(Xte), 40))
    for cfg in CFGS.values():
        for rep, a, b in folds:
            for j in range(40):
                m = ~Ptr[a, j]; y = Ytr[a, j][m]
                mdl = lgb.LGBMClassifier(objective="binary", verbose=-1, random_state=0, n_jobs=8, **cfg).fit(Xtr[a][m], y)
                oof[b, j] += mdl.predict_proba(Xtr[b])[:, 1] / (NREP * len(CFGS)); pv[:, j] += mdl.predict_proba(Xva)[:, 1] / (len(folds) * len(CFGS)); pt[:, j] += mdl.predict_proba(Xte)[:, 1] / (len(folds) * len(CFGS))
    a_oof, a_val = aps(oof, Ytr, Ptr), aps(pv, Yn[va], Pn[va])
    print(f"{name:46s} OOF mAP={a_oof.mean():.4f} (acc {a_oof[ai].mean():.3f} / term {a_oof[ti].mean():.3f} / chain {a_oof[ci].mean():.3f}) | val mAP={a_val.mean():.4f} AUROC={auc(pv, Yn[va], Pn[va]):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return oof, pv, pt, a_oof, a_val

X = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(pats.PATIENT)
res = {}
res["clean"] = run(X, "clean structured tables (primary)")
if (HERE / "clean" / "X_noqol.parquet").exists():
    Xq = pd.read_parquet(HERE / "clean" / "X_noqol.parquet").reindex(pats.PATIENT); res["noqol"] = run(Xq, "sensitivity: without QALY/DALY/QOLS")
# for the record only: the leaked one-bit genomics channel
gid = set(os.path.basename(f).split("_dna")[0] for f in glob.glob(str(DATA / "genomics" / "*_dna.csv")))
Xd = X.copy(); Xd["has_dna_LEAK"] = pats.PATIENT.isin(gid).values.astype(np.float32)
res["dna"] = run(Xd, "NOT USED: + 'has DNA file' (coherence leak)")

oof, pv, pt, a_oof, a_val = res["clean"]
def group_avg(p, Pm):
    p = p.copy()
    for grp in GROUPS:
        jj = [codes.index(c) for c in grp]; m = p[:, jj].mean(1, keepdims=True); p[:, jj] = np.where(Pm[:, jj], 0, m)
    return p
final_te = group_avg(np.where(Pn[te], 0, pt), Pn[te]); final_val = group_avg(np.where(Pn[va], 0, pv), Pn[va])
pd.DataFrame(final_te, index=pd.Index(pats.PATIENT.values[te], name="patient_id"), columns=codes).to_csv(ROOT / "predictions_clean.csv")
pd.DataFrame(final_val, index=pd.Index(pats.PATIENT.values[va], name="patient_id"), columns=codes).to_csv(HERE / "clean" / "val_predictions_clean.csv")
np.save(HERE / "clean" / "oof.npy", oof); np.save(HERE / "clean" / "val.npy", pv); np.save(HERE / "clean" / "test.npy", pt)
per = pd.DataFrame({"cond": [desc[c][:34] for c in codes], "oof_clean": res["clean"][3], "val_clean": res["clean"][4], "oof_noqol": res.get("noqol", [None] * 5)[3] if "noqol" in res else np.nan, "oof_with_dna_leak": res["dna"][3]}, index=codes).round(3)
pd.set_option("display.width", 200); print(per.to_string()); per.to_csv(HERE / "clean" / "per_condition_clean.csv")
json.dump({k: {"oof_mAP": float(v[3].mean()), "val_mAP": float(v[4].mean())} for k, v in res.items()}, open(HERE / "clean" / "summary_clean.json", "w"), indent=1)
print("wrote predictions_clean.csv", f"({time.time()-t0:.0f}s)")
