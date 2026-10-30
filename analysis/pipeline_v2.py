"""Iteration 2: OOF-driven tabular pipeline + blend with the project's transformer.

Steps (all decisions by 5-fold x 2-repeat OOF on the 2791 official train patients; val reported, never used to choose):
  1. richer hand-crafted history features (our own representation of the raw tables)
  2. per-target LightGBM (2 configs), fold-ensembled for val/test
  3. death-in-window model and a two-stage death x cause model for the 10 death-aligned targets
  4. per-target stacking weight between GBM and an age/sex logistic model (chosen on OOF)
  5. blend with the project's strict C0R0 transformer (3 seeds, val/test predictions already on disk)
  6. duplicate-target averaging; write predictions_v2.csv (test) + val_predictions_v2.csv
"""
import json, time, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from feats import build_features
from ehr_load import load
warnings.filterwarnings("ignore")
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "v2"; OUT.mkdir(exist_ok=True, parents=True)
NREP, NFOLD = 2, 5

# ---------------------------------------------------------------- 1. features
d, X = build_features()
pats, events, codes, targets = d["pats"], d["events"], d["codes"], d["targets"]
pidx = pats.set_index("PATIENT"); ids = pats.PATIENT.values
desc = dict(zip(targets.CODE, targets.DESCRIPTION)); short = {c: desc[c][:34] for c in codes}
# extra numeric-trend features: std and count of measurements, last minus earlier mean (5y window)
pre = events[events.pre & (events.kind == "OBS") & events.num.notna()].copy()
pre["yb"] = (pre.PATIENT.map(pidx.anchor) - pre.date).dt.days / 365.25
nk = [c.split("last_", 1)[1] for c in X.columns if c.startswith("last_")]
pre = pre[pre.code.isin(nk)].sort_values("date")
g = pre.groupby(["PATIENT", "code"]).num
extra = [g.std().unstack().reindex(index=ids, columns=nk).add_prefix("std_"),
         g.size().unstack().reindex(index=ids, columns=nk).fillna(0).add_prefix("nobs_"),
         (g.last() - pre[pre.yb > 2].groupby(["PATIENT", "code"]).num.mean()).unstack().reindex(index=ids, columns=nk).add_prefix("trend_")]
# encounter rates per year of history, by class
enc = events[events.pre & (events.kind == "ENC")]
rates = enc.groupby(["PATIENT", "encclass"]).size().unstack(fill_value=0).reindex(index=ids).fillna(0)
hist_years = X["hist_years"].clip(lower=1.0)
rates = rates.div(hist_years, axis=0).add_prefix("encrate_")
X = pd.concat([X] + extra + [rates], axis=1).astype(np.float32)
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Y = d["Y"].values; P = d["P"].values.astype(bool); died = pats.died.values.astype(int)
Xtr, Xva, Xte = X.values[tr], X.values[va], X.values[te]
Ytr, Ptr, Yva, Pva, Pte = Y[tr], P[tr], Y[va], P[va], P[te]
print(f"features {X.shape}  ({time.time()-t0:.0f}s)", flush=True)

TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
GROUPS = [["162573006", "424132000", "254637007"], ["22325002", "723857007"]]   # identical-label groups (Jaccard 0.9-1.0)
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]

def aps(p, Yv, Pv):
    p = np.where(Pv, 0, p)
    return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])

def report(name, p_oof, p_val):
    a_oof, a_val = aps(p_oof, Ytr, Ptr), aps(p_val, Yva, Pva)
    ti = [codes.index(c) for c in TERMINAL]; ai = [codes.index(c) for c in ACCIDENT]; ci = [j for j in range(40) if j not in ti and j not in ai]
    print(f"{name:52s} OOF mAP={a_oof.mean():.4f} (acc {a_oof[ai].mean():.3f} / term {a_oof[ti].mean():.3f} / chain {a_oof[ci].mean():.3f}) | val mAP={a_val.mean():.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return a_oof, a_val

folds = [(rep, a, b) for rep in range(NREP) for a, b in KFold(NFOLD, shuffle=True, random_state=rep).split(Xtr)]

def cv_fit(make_model, X_tr, y_all, mask_all, X_val, X_te, rows=None):
    """Per-target CV: returns oof (n_tr x 40), val and test predictions averaged over fold models."""
    oof = np.zeros((len(X_tr), 40)); pv = np.zeros((len(X_val), 40)); pt = np.zeros((len(X_te), 40))
    for rep, a, b in folds:
        for j in range(40):
            sel = a if rows is None else a[rows[a]]
            m = mask_all[sel, j]; y = y_all[sel, j][m]
            if y.sum() < 3 or y.sum() == len(y):
                oof[b, j] += y.mean() / NREP; pv[:, j] += y.mean() / len(folds); pt[:, j] += y.mean() / len(folds); continue
            mdl = make_model().fit(X_tr[sel][m], y)
            oof[b, j] += mdl.predict_proba(X_tr[b])[:, 1] / NREP
            pv[:, j] += mdl.predict_proba(X_val)[:, 1] / len(folds); pt[:, j] += mdl.predict_proba(X_te)[:, 1] / len(folds)
    return oof, pv, pt

# ---------------------------------------------------------------- 2. GBM (two configs, averaged)
CFGS = {"base": dict(learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=350),
        "shallow": dict(learning_rate=0.02, num_leaves=4, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.2, reg_lambda=10.0, n_estimators=600)}
atrisk = ~Ptr
gbm = {}
for name, cfg in CFGS.items():
    mk = lambda cfg=cfg: lgb.LGBMClassifier(objective="binary", verbose=-1, random_state=0, n_jobs=8, **cfg)
    gbm[name] = cv_fit(mk, Xtr, Ytr, atrisk, Xva, Xte)
    report(f"GBM {name}", gbm[name][0], gbm[name][1])
g_oof = np.mean([v[0] for v in gbm.values()], 0); g_val = np.mean([v[1] for v in gbm.values()], 0); g_te = np.mean([v[2] for v in gbm.values()], 0)
report("GBM 2-config average", g_oof, g_val)

# ---------------------------------------------------------------- 3. simple age/sex model
def simple_feats(mask):
    a = pats.age.values[mask]; return np.c_[a, a ** 2, pats.female.values[mask]]
sc = StandardScaler().fit(simple_feats(tr)); S_tr, S_va, S_te = sc.transform(simple_feats(tr)), sc.transform(simple_feats(va)), sc.transform(simple_feats(te))
mk_lr = lambda: LogisticRegression(C=1.0, max_iter=2000)
s_oof, s_val, s_te = cv_fit(mk_lr, S_tr, Ytr, atrisk, S_va, S_te)
report("age/sex logistic", s_oof, s_val)

# ---------------------------------------------------------------- 4. death model + two-stage for terminal targets
mk_d = lambda: lgb.LGBMClassifier(objective="binary", verbose=-1, random_state=0, n_jobs=8, **CFGS["base"])
d_oof = np.zeros(len(Xtr)); d_val = np.zeros(len(Xva)); d_te = np.zeros(len(Xte))
for rep, a, b in folds:
    mdl = mk_d().fit(Xtr[a], died[tr][a]); d_oof[b] += mdl.predict_proba(Xtr[b])[:, 1] / NREP
    d_val += mdl.predict_proba(Xva)[:, 1] / len(folds); d_te += mdl.predict_proba(Xte)[:, 1] / len(folds)
print(f"death model: OOF AUROC={roc_auc_score(died[tr], d_oof):.3f}  val AUROC={roc_auc_score(died[va], d_val):.3f}", flush=True)
# stage 2: P(Y_k | H, D=1) trained on dead training patients only
dead_rows = died[tr].astype(bool)
c_oof, c_val, c_te = cv_fit(lambda: lgb.LGBMClassifier(objective="binary", verbose=-1, random_state=0, n_jobs=8, **CFGS["shallow"]), Xtr, Ytr, atrisk, Xva, Xte, rows=dead_rows)
ts_oof, ts_val, ts_te = g_oof.copy(), g_val.copy(), g_te.copy()
for c in TERMINAL:
    j = codes.index(c); ts_oof[:, j] = d_oof * c_oof[:, j]; ts_val[:, j] = d_val * c_val[:, j]; ts_te[:, j] = d_te * c_te[:, j]
report("two-stage death x cause (terminal targets)", ts_oof, ts_val)
# average of direct GBM and two-stage on terminal targets
avg_oof, avg_val, avg_te = (g_oof + ts_oof) / 2, (g_val + ts_val) / 2, (g_te + ts_te) / 2
report("GBM + two-stage average", avg_oof, avg_val)

# ---------------------------------------------------------------- 5. per-target stacking with the simple model (weights chosen on OOF)
W = np.linspace(0, 1, 11)
best_w = np.zeros(40); st_oof = np.zeros_like(g_oof); st_val = np.zeros_like(g_val); st_te = np.zeros_like(g_te)
base_oof, base_val, base_te = avg_oof, avg_val, avg_te
for j in range(40):
    scores = [average_precision_score(Ytr[:, j], np.where(Ptr[:, j], 0, w * base_oof[:, j] + (1 - w) * s_oof[:, j])) for w in W]
    w = W[int(np.argmax(scores))]; best_w[j] = w
    st_oof[:, j] = w * base_oof[:, j] + (1 - w) * s_oof[:, j]; st_val[:, j] = w * base_val[:, j] + (1 - w) * s_val[:, j]; st_te[:, j] = w * base_te[:, j] + (1 - w) * s_te[:, j]
print("stacking weights on GBM side:", dict(zip([short[c][:18] for c in codes], best_w.round(1))), flush=True)
a_oof, a_val = report("tabular final (stacked, in-sample OOF weights -> optimistic)", st_oof, st_val)

# ---------------------------------------------------------------- 6. blend with the project's transformer (strict C0R0, 3 seeds)
val_ids = ids[va]; te_ids = ids[te]
def load_tf(split):
    fs = sorted(glob.glob(str(ROOT / "artifacts" / "experimental_audit_strict" / "runs" / f"C0R0-seed*-repeat-a" / f"{split}_predictions.csv")))
    return np.mean([pd.read_csv(f, dtype={"patient_id": str}).set_index("patient_id").reindex(val_ids if split == "val" else te_ids)[codes].values for f in fs], 0)
tf_val, tf_te = load_tf("val"), load_tf("test")
def group_avg(p, Pv):
    p = p.copy()
    for grp in GROUPS:
        jj = [codes.index(c) for c in grp]; m = p[:, jj].mean(1, keepdims=True); p[:, jj] = np.where(Pv[:, jj], 0, m)
    return p
cands = {"tabular final": (st_val, st_te), "transformer C0R0 x3": (tf_val, tf_te),
         "blend 0.5 tabular + 0.5 transformer": ((st_val + tf_val) / 2, (st_te + tf_te) / 2),
         "blend 0.6 tabular + 0.4 transformer": (0.6 * st_val + 0.4 * tf_val, 0.6 * st_te + 0.4 * tf_te)}
for name, (pv, pt) in cands.items():
    a = aps(pv, Yva, Pva); ag = aps(group_avg(pv, Pva), Yva, Pva)
    print(f"{name:40s} val mAP={a.mean():.4f}  (+dup-group avg {ag.mean():.4f})  macroAUROC={np.mean([roc_auc_score(Yva[:, j], np.where(Pva[:, j], 0, pv[:, j])) for j in range(40)]):.4f}", flush=True)

# final: 50/50 blend (pre-registered in the diagnosis doc), duplicate-group averaging, prior masking
final_val = group_avg(np.where(Pva, 0, (st_val + tf_val) / 2), Pva); final_te = group_avg(np.where(Pte, 0, (st_te + tf_te) / 2), Pte)
pd.DataFrame(final_te, index=pd.Index(te_ids, name="patient_id"), columns=codes).to_csv(ROOT / "predictions_v2.csv")
pd.DataFrame(final_val, index=pd.Index(val_ids, name="patient_id"), columns=codes).to_csv(OUT / "val_predictions_v2.csv")
np.save(OUT / "tabular_oof.npy", st_oof); np.save(OUT / "tabular_val.npy", st_val); np.save(OUT / "tabular_test.npy", st_te)
per = pd.DataFrame({"cond": [short[c] for c in codes], "val_pos": Yva.sum(0), "oof_gbm": aps(g_oof, Ytr, Ptr), "oof_twostage": aps(ts_oof, Ytr, Ptr), "oof_simple": aps(s_oof, Ytr, Ptr),
                    "oof_tabular_final": a_oof, "stack_w": best_w, "val_tabular": a_val, "val_transformer": aps(tf_val, Yva, Pva), "val_final_blend": aps(final_val, Yva, Pva)}, index=codes).round(3)
per.to_csv(OUT / "per_condition_v2.csv"); pd.set_option("display.width", 250); print(per.to_string())
json.dump({"features": int(X.shape[1]), "oof_final_tabular": float(a_oof.mean()), "val_final_blend": float(aps(final_val, Yva, Pva).mean())}, open(OUT / "summary_v2.json", "w"), indent=1)
print("wrote predictions_v2.csv", f"({time.time()-t0:.0f}s)")
