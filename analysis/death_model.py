"""Cross-fitted P(death in the 5-year window | pre-anchor history): LightGBM on the clean feature matrix, 5 patient
folds (OOF for train; fold-average for val/test). 'Died in window' is a label derived from the training patients' own
future, exactly like the 40 targets, so it is legal to learn; it is used only as an input to the cost-residual features
and as a diagnostic. Writes analysis/stack/pdeath_{oof,val,test}.npy."""
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from ehr_load import load
t0 = time.time(); HERE = Path(__file__).resolve().parent; OUT = HERE / "stack"; OUT.mkdir(exist_ok=True)
d = load(); pats = d["pats"]; ids = {s: pats[pats.split == s].PATIENT.values for s in ("train", "val", "test")}
died = pats.set_index("PATIENT").died.reindex(ids["train"]).values.astype(int)
X = pd.read_parquet(HERE / "clean" / "X.parquet"); Xs = {s: X.reindex(ids[s]).values for s in ids}
cfg = dict(objective="binary", verbose=-1, n_jobs=8, learning_rate=0.03, n_estimators=300, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0)
pD = {"train": np.zeros(len(ids["train"])), "val": np.zeros(len(ids["val"])), "test": np.zeros(len(ids["test"]))}
for a, b in KFold(5, shuffle=True, random_state=11).split(Xs["train"]):
    m = lgb.LGBMClassifier(**cfg).fit(Xs["train"][a], died[a]); pD["train"][b] = m.predict_proba(Xs["train"][b])[:, 1]
    pD["val"] += m.predict_proba(Xs["val"])[:, 1] / 5; pD["test"] += m.predict_proba(Xs["test"])[:, 1] / 5
print(f"death model OOF AUROC {roc_auc_score(died, pD['train']):.4f} | died share train {died.mean():.3f} [{time.time()-t0:.0f}s]")
for s, k in (("train", "oof"), ("val", "val"), ("test", "test")): np.save(OUT / f"pdeath_{k}.npy", pD[s])
