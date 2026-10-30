"""Iteration 3: competing-risks (death-pathway) structured output for the death-aligned targets.

  p_k(H) = P(D=0|H) * r_k0  +  P(D=1|H) * sum_pi P(pi | H, D=1) * T[k, pi]
  pi in {none, lung, prostate, cardiac, pneumonia, stroke, colorectal, alzheimer}; T[k,pi] = empirical P(Y_k | pi, D=1) on train.
Compared on 5-fold x 2 OOF against (a) direct per-target GBM and (b) independent two-stage sigmoids (v2).
Everything else (features, folds, stacking, blend) mirrors pipeline_v2.py.
"""
import json, time, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from feats import build_features
warnings.filterwarnings("ignore")
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "analysis" / "v3"; OUT.mkdir(exist_ok=True, parents=True)
NREP, NFOLD = 2, 5

d, X = build_features()
pats, events, codes, targets = d["pats"], d["events"], d["codes"], d["targets"]
pidx = pats.set_index("PATIENT"); ids = pats.PATIENT.values
desc = dict(zip(targets.CODE, targets.DESCRIPTION)); short = {c: desc[c][:34] for c in codes}
pre = events[events.pre & (events.kind == "OBS") & events.num.notna()].copy()
pre["yb"] = (pre.PATIENT.map(pidx.anchor) - pre.date).dt.days / 365.25
nk = [c.split("last_", 1)[1] for c in X.columns if c.startswith("last_")]
pre = pre[pre.code.isin(nk)].sort_values("date"); g = pre.groupby(["PATIENT", "code"]).num
extra = [g.std().unstack().reindex(index=ids, columns=nk).add_prefix("std_"), g.size().unstack().reindex(index=ids, columns=nk).fillna(0).add_prefix("nobs_"),
         (g.last() - pre[pre.yb > 2].groupby(["PATIENT", "code"]).num.mean()).unstack().reindex(index=ids, columns=nk).add_prefix("trend_")]
enc = events[events.pre & (events.kind == "ENC")]
rates = enc.groupby(["PATIENT", "encclass"]).size().unstack(fill_value=0).reindex(index=ids).fillna(0).div(X["hist_years"].clip(lower=1.0), axis=0).add_prefix("encrate_")
X = pd.concat([X] + extra + [rates], axis=1).astype(np.float32)
tr = (pats.split == "train").values; va = (pats.split == "val").values; te = (pats.split == "test").values
Y = d["Y"].values; P = d["P"].values.astype(bool); died = pats.died.values.astype(int)
Xtr, Xva, Xte = X.values[tr], X.values[va], X.values[te]; Ytr, Ptr, Yva, Pva, Pte = Y[tr], P[tr], Y[va], P[va], P[te]
print(f"features {X.shape} ({time.time()-t0:.0f}s)", flush=True)

PATH = {"lung": ["162573006", "424132000", "254637007"], "prostate": ["126906006", "92691004", "314994000"], "cardiac": ["88805009", "22298006"],
        "pneumonia": ["233604007"], "stroke": ["230690007"], "colorectal": ["68496003", "713197008"], "alzheimer": ["26929004", "22325002", "723857007"]}
PNAMES = ["none"] + list(PATH)
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
STRUCT = sorted(set(sum(PATH.values(), [])))      # all targets covered by a pathway (15)
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
GROUPS = [["162573006", "424132000", "254637007"], ["22325002", "723857007"]]

# primary pathway label: among a dead patient's window pathways, the one whose terminal diagnosis is closest to the record end
Ft = d["first_targets"].loc[ids]; endv = pidx["end"].reindex(ids)
def primary_pathway(i):
    best, bestd = 0, -1
    for p, (name, cs) in enumerate(PATH.items(), start=1):
        if Y[i, [codes.index(c) for c in cs]].max() == 0: continue
        dt = max((Ft.iloc[i][c] for c in cs if Y[i, codes.index(c)] == 1), default=pd.NaT)
        if pd.isna(dt): continue
        if dt > bestd if bestd != -1 else True:
            if bestd == -1 or dt > bestd: best, bestd = p, dt
    return best
pi = np.array([primary_pathway(i) if died[i] else 0 for i in range(len(ids))])
print("primary pathway counts (train, dead):", dict(zip(PNAMES, np.bincount(pi[tr & (died == 1)], minlength=len(PNAMES)))), flush=True)

def aps(p, Yv, Pv):
    p = np.where(Pv, 0, p); return np.array([average_precision_score(Yv[:, j], p[:, j]) if Yv[:, j].sum() else 0 for j in range(40)])
def report(name, p_oof, p_val):
    a_oof, a_val = aps(p_oof, Ytr, Ptr), aps(p_val, Yva, Pva)
    ti = [codes.index(c) for c in TERMINAL]; si = [codes.index(c) for c in STRUCT]; ai = [codes.index(c) for c in ACCIDENT]
    print(f"{name:48s} OOF mAP={a_oof.mean():.4f} (terminal10 {a_oof[ti].mean():.3f} / pathway15 {a_oof[si].mean():.3f}) | val mAP={a_val.mean():.4f} (terminal10 {a_val[ti].mean():.3f})  [{time.time()-t0:.0f}s]", flush=True)
    return a_oof, a_val

folds = [(rep, a, b) for rep in range(NREP) for a, b in KFold(NFOLD, shuffle=True, random_state=rep).split(Xtr)]
CFG = dict(learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=350)
CFG_S = dict(learning_rate=0.02, num_leaves=4, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.2, reg_lambda=10.0, n_estimators=600)
mk = lambda cfg: lgb.LGBMClassifier(objective="binary", verbose=-1, random_state=0, n_jobs=8, **cfg)

def cv_fit(make_model, X_tr, y_all, mask_all, X_val, X_te, rows=None):
    oof = np.zeros((len(X_tr), 40)); pv = np.zeros((len(X_val), 40)); pt = np.zeros((len(X_te), 40))
    for rep, a, b in folds:
        for j in range(40):
            sel = a if rows is None else a[rows[a]]; m = mask_all[sel, j]; y = y_all[sel, j][m]
            if y.sum() < 3 or y.sum() == len(y):
                oof[b, j] += y.mean() / NREP; pv[:, j] += y.mean() / len(folds); pt[:, j] += y.mean() / len(folds); continue
            mdl = make_model().fit(X_tr[sel][m], y)
            oof[b, j] += mdl.predict_proba(X_tr[b])[:, 1] / NREP; pv[:, j] += mdl.predict_proba(X_val)[:, 1] / len(folds); pt[:, j] += mdl.predict_proba(X_te)[:, 1] / len(folds)
    return oof, pv, pt

# (a) direct GBM, two configs averaged (as in v2)
g1 = cv_fit(lambda: mk(CFG), Xtr, Ytr, ~Ptr, Xva, Xte); g2 = cv_fit(lambda: mk(CFG_S), Xtr, Ytr, ~Ptr, Xva, Xte)
g_oof, g_val, g_te = [(x + y) / 2 for x, y in zip(g1, g2)]
report("(a) direct GBM 2-config", g_oof, g_val)

# death gate
d_oof = np.zeros(len(Xtr)); d_val = np.zeros(len(Xva)); d_te = np.zeros(len(Xte))
for rep, a, b in folds:
    mdl = mk(CFG).fit(Xtr[a], died[tr][a]); d_oof[b] += mdl.predict_proba(Xtr[b])[:, 1] / NREP; d_val += mdl.predict_proba(Xva)[:, 1] / len(folds); d_te += mdl.predict_proba(Xte)[:, 1] / len(folds)
print(f"death gate OOF AUROC={roc_auc_score(died[tr], d_oof):.3f} val AUROC={roc_auc_score(died[va], d_val):.3f}", flush=True)

# (b) independent two-stage sigmoids on the dead
dead_tr = died[tr].astype(bool)
c_oof, c_val, c_te = cv_fit(lambda: mk(CFG_S), Xtr, Ytr, ~Ptr, Xva, Xte, rows=dead_tr)
r0 = np.array([Ytr[(~dead_tr) & ~Ptr[:, j], j].mean() for j in range(40)])        # P(Y_k | D=0) empirical
ind_oof, ind_val, ind_te = g_oof.copy(), g_val.copy(), g_te.copy()
for c in STRUCT:
    j = codes.index(c)
    ind_oof[:, j] = d_oof * c_oof[:, j] + (1 - d_oof) * r0[j]; ind_val[:, j] = d_val * c_val[:, j] + (1 - d_val) * r0[j]; ind_te[:, j] = d_te * c_te[:, j] + (1 - d_te) * r0[j]
report("(b) two-stage independent sigmoids (15 pathway targets)", ind_oof, ind_val)

# (c) competing-risks pathway softmax on the dead + empirical T[k, pi]
K = len(PNAMES); pi_tr = pi[tr]
m_oof = np.zeros((len(Xtr), K)); m_val = np.zeros((len(Xva), K)); m_te = np.zeros((len(Xte), K))
T_oof = np.zeros((len(Xtr), 40))
for rep, a, b in folds:
    sel = a[dead_tr[a]]
    mdl = lgb.LGBMClassifier(objective="multiclass", num_class=K, verbose=-1, random_state=0, n_jobs=8, **CFG_S).fit(Xtr[sel], pi_tr[sel])
    pb, pv_, pt_ = mdl.predict_proba(Xtr[b]), mdl.predict_proba(Xva), mdl.predict_proba(Xte)
    # empirical P(Y_k | pi, D=1, at risk) from this fold's dead training patients
    T = np.zeros((K, 40))
    for p in range(K):
        rows = sel[pi_tr[sel] == p]
        for j in range(40):
            rr = rows[~Ptr[rows, j]]; T[p, j] = (Ytr[rr, j].sum() + 0.5) / (len(rr) + 1.0)
    m_oof[b] += pb / NREP; m_val += pv_ / len(folds); m_te += pt_ / len(folds)
    T_oof[b] += (pb @ T) / NREP
    if rep == 0 and b[0] == folds[0][2][0]: T_val, T_te = pv_ @ T / len(folds), pt_ @ T / len(folds)
    else: T_val += pv_ @ T / len(folds); T_te += pt_ @ T / len(folds)
cr_oof, cr_val, cr_te = g_oof.copy(), g_val.copy(), g_te.copy()
for c in STRUCT:
    j = codes.index(c)
    cr_oof[:, j] = d_oof * T_oof[:, j] + (1 - d_oof) * r0[j]; cr_val[:, j] = d_val * T_val[:, j] + (1 - d_val) * r0[j]; cr_te[:, j] = d_te * T_te[:, j] + (1 - d_te) * r0[j]
report("(c) competing-risks pathway softmax", cr_oof, cr_val)
report("(a)+(c) average", (g_oof + cr_oof) / 2, (g_val + cr_val) / 2)
report("(a)+(b)+(c) average", (g_oof + ind_oof + cr_oof) / 3, (g_val + ind_val + cr_val) / 3)
pm = m_oof.argmax(1); print("pathway softmax OOF accuracy on dead train:", round(float((pm[dead_tr] == pi_tr[dead_tr]).mean()), 3), "| majority baseline:", round(float(np.bincount(pi_tr[dead_tr]).max() / dead_tr.sum()), 3), flush=True)

per = pd.DataFrame({"cond": [short[c] for c in codes], "oof_direct": aps(g_oof, Ytr, Ptr), "oof_twostage": aps(ind_oof, Ytr, Ptr), "oof_pathway": aps(cr_oof, Ytr, Ptr), "oof_abc": aps((g_oof + ind_oof + cr_oof) / 3, Ytr, Ptr),
                    "val_direct": aps(g_val, Yva, Pva), "val_pathway": aps(cr_val, Yva, Pva), "val_abc": aps((g_val + ind_val + cr_val) / 3, Yva, Pva)}, index=codes).round(3)
pd.set_option("display.width", 250); print(per.loc[STRUCT].to_string())
per.to_csv(OUT / "per_condition_v3.csv")

# final tabular = (a)+(b)+(c) average, then per-target shrinkage toward age/sex model (weights on OOF), as in v2
base_oof, base_val, base_te = (g_oof + ind_oof + cr_oof) / 3, (g_val + ind_val + cr_val) / 3, (g_te + ind_te + cr_te) / 3
def simple_feats(mask):
    a = pats.age.values[mask]; return np.c_[a, a ** 2, pats.female.values[mask]]
sc = StandardScaler().fit(simple_feats(tr)); S = [sc.transform(simple_feats(m)) for m in (tr, va, te)]
s_oof, s_val, s_te = cv_fit(lambda: LogisticRegression(C=1.0, max_iter=2000), S[0], Ytr, ~Ptr, S[1], S[2])
W = np.linspace(0, 1, 11); st = [np.zeros_like(x) for x in (base_oof, base_val, base_te)]; wbest = np.zeros(40)
for j in range(40):
    sc_ = [average_precision_score(Ytr[:, j], np.where(Ptr[:, j], 0, w * base_oof[:, j] + (1 - w) * s_oof[:, j])) for w in W]; w = W[int(np.argmax(sc_))]; wbest[j] = w
    for k, (bse, sm) in enumerate(((base_oof, s_oof), (base_val, s_val), (base_te, s_te))): st[k][:, j] = w * bse[:, j] + (1 - w) * sm[:, j]
a_oof, a_val = report("tabular v3 final (stacked)", st[0], st[1])
np.save(OUT / "tabular_oof.npy", st[0]); np.save(OUT / "tabular_val.npy", st[1]); np.save(OUT / "tabular_test.npy", st[2])

# blend with transformer and write predictions_v3.csv
val_ids, te_ids = ids[va], ids[te]
def load_tf(split):
    fs = sorted(glob.glob(str(ROOT / "artifacts" / "experimental_audit_strict" / "runs" / "C0R0-seed*-repeat-a" / f"{split}_predictions.csv")))
    return np.mean([pd.read_csv(f, dtype={"patient_id": str}).set_index("patient_id").reindex(val_ids if split == "val" else te_ids)[codes].values for f in fs], 0)
tf_val, tf_te = load_tf("val"), load_tf("test")
def group_avg(p, Pv):
    p = p.copy()
    for grp in GROUPS:
        jj = [codes.index(c) for c in grp]; mm = p[:, jj].mean(1, keepdims=True); p[:, jj] = np.where(Pv[:, jj], 0, mm)
    return p
for name, pv in (("tabular v3", st[1]), ("transformer x3", tf_val), ("blend 50/50", (st[1] + tf_val) / 2), ("blend 50/50 + dup-avg", group_avg((st[1] + tf_val) / 2, Pva))):
    a = aps(pv, Yva, Pva); print(f"{name:28s} val mAP={a.mean():.4f}  macroAUROC={np.mean([roc_auc_score(Yva[:, j], np.where(Pva[:, j], 0, pv[:, j])) for j in range(40)]):.4f}", flush=True)
final_val = group_avg(np.where(Pva, 0, (st[1] + tf_val) / 2), Pva); final_te = group_avg(np.where(Pte, 0, (st[2] + tf_te) / 2), Pte)
pd.DataFrame(final_te, index=pd.Index(te_ids, name="patient_id"), columns=codes).to_csv(ROOT / "predictions_v3.csv")
pd.DataFrame(final_val, index=pd.Index(val_ids, name="patient_id"), columns=codes).to_csv(OUT / "val_predictions_v3.csv")
json.dump({"oof_tabular_v3": float(a_oof.mean()), "val_blend_v3": float(aps(final_val, Yva, Pva).mean()), "stack_w": wbest.tolist()}, open(OUT / "summary_v3.json", "w"), indent=1)
print("wrote predictions_v3.csv", f"({time.time()-t0:.0f}s)")
