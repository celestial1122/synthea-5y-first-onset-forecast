"""Per-target LightGBM on genomics/imaging features + static (age, sex, prior_*) only — an undiluted candidate for the
targets whose signal lives in the DNA files. 5-fold patient CV on official train rows; fit-on-all for val/test.
Outputs gstack/gen_only_{oof,val,test}.npy (candidate layout)."""
import numpy as np, pandas as pd, lightgbm as lgb, time, json, os
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"; OUT = HERE / "gstack"; OUT.mkdir(exist_ok=True)
from ehr_load import load as _load; _pats = _load()["pats"]
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); codes = list(M["codes"]); off = (M["k"] == 0)
ids = {s: _pats[_pats.split == s].PATIENT.values for s in ("train", "val", "test")}
EXPORT = (Path(os.environ["EXPORT_DIR"]) / "C") if os.environ.get("EXPORT_DIR") else None; WANDB = bool(os.environ.get("WANDB_PROJECT"))
if EXPORT: EXPORT.mkdir(parents=True, exist_ok=True)
if WANDB: import wandb
def lab(split, order):
    m = off & (M["split"] == split); pos = pd.Series(np.arange(m.sum()), index=M["pid"][m].astype(str)).reindex(order).values.astype(int)
    return M["y"][m][pos], M["prior"][m][pos].astype(bool)
Y, P = {}, {}
for s in ("train", "val", "test"): Y[s], P[s] = lab(s, ids[s])
X = pd.read_parquet(HERE / "clean/X.parquet"); import sys; COST = any(a in sys.argv[1:] for a in ("cost", "cost2", "cost3")); COST2 = "cost2" in sys.argv[1:]; COST3 = "cost3" in sys.argv[1:]; SEED = next((int(a[4:]) for a in sys.argv[1:] if a.startswith("seed")), 0)
NAME = ("genc3_only" if COST3 else "genc2_only" if COST2 else ("genc_only" if COST else "gen_only")) + (f"_s{SEED}" if SEED else "")
G = pd.read_parquet(HERE / "clean/genomics_features.parquet")
if COST: G = G.join(pd.read_parquet(HERE / ("clean/cost_features_v3.parquet" if COST3 else "clean/cost_features_v2.parquet" if COST2 else "clean/cost_features.parquet")), how="left")
static = ["age", "female"] + [c for c in X.columns if c.startswith("prior_")]
F = {s: np.column_stack([X.reindex(ids[s])[static].values.astype(float), G.reindex(ids[s]).values.astype(float)]) for s in ids}
params = dict(objective="binary", learning_rate=0.03, num_leaves=6, min_child_samples=10, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=8, seed=SEED)
oof = np.zeros((len(ids["train"]), 40)); pv = np.zeros((len(ids["val"]), 40)); pt = np.zeros((len(ids["test"]), 40)); res = []
feat_names = static + list(G.columns)
if EXPORT: json.dump({"member": "C", "columns": feat_names, "params": params, "rounds": 300, "cost_mode": "cost2" if COST2 else "cost3" if COST3 else "cost" if COST else "none"}, open(EXPORT / "meta.json", "w"), indent=1)
if WANDB:
    wandb.init(project=os.environ["WANDB_PROJECT"], name=f"memberC_history_free_{NAME}", config=dict(member="C", model="lightgbm on age/sex/prior + genomics + cost (no history tokens)", rounds=300, n_feats=len(feat_names), **params))
    wandb.define_metric("round"); wandb.define_metric("curves/*", step_metric="round")
curves = []
for j, c in enumerate(codes):
    at = ~P["train"][:, j]; y = Y["train"][at, j]; Xj = F["train"][at]; o = np.zeros(at.sum())
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xj):
        b = lgb.train(params, lgb.Dataset(Xj[tr], y[tr]), 300); o[te] = b.predict(Xj[te])
    if WANDB:
        ev = {}; b = lgb.train(params, lgb.Dataset(Xj, y), 300, valid_sets=[lgb.Dataset(Xj, y), lgb.Dataset(F["val"][~P["val"][:, j]], Y["val"][~P["val"][:, j], j])], valid_names=["train", "val"], callbacks=[lgb.record_evaluation(ev)])
        curves.append((ev["train"]["binary_logloss"], ev["val"]["binary_logloss"]))
    else: b = lgb.train(params, lgb.Dataset(Xj, y), 300)
    if EXPORT: b.save_model(str(EXPORT / f"{c}.txt"))
    oof[at, j] = o; pv[:, j] = b.predict(F["val"]); pt[:, j] = b.predict(F["test"])
    res.append((c, ap(y, o), ap(Y["val"][~P["val"][:, j], j], pv[~P["val"][:, j], j])))
R = pd.DataFrame(res, columns=["code", "oof", "val"]); print(R.round(3).to_string(index=False))
print(f"{NAME}: OOF mAP {R.oof.mean():.4f} | val {R.val.mean():.4f}  [{time.time()-t0:.0f}s]")
if WANDB:
    tr_c = np.mean([c_[0] for c_ in curves], 0); va_c = np.mean([c_[1] for c_ in curves], 0)
    for r in range(len(tr_c)): wandb.log({"round": r + 1, "curves/train_logloss": float(tr_c[r]), "curves/val_logloss": float(va_c[r])})
    wandb.summary.update({"oof_mAP": float(R.oof.mean()), "val_mAP": float(R.val.mean())}); wandb.log({"per_target": wandb.Table(dataframe=R)}); wandb.finish()
for s, a in (("oof", oof), ("val", pv), ("test", pt)): np.save(OUT / f"{NAME}_{s}.npy", a)
