"""Diagnose the data-generating process: anchor artifact, death, chains, predictability ceilings."""
import json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from ehr_load import load, DATA
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)

d = load()
pats, events, Y, P, codes, targets = d["pats"], d["events"], d["Y"], d["P"], d["codes"], d["targets"]
desc = dict(zip(targets.CODE, targets.DESCRIPTION))
short = {c: desc[c][:34] for c in codes}
tv = pats[pats.split != "test"].copy()
train = pats[pats.split == "train"]; val = pats[pats.split == "val"]; test = pats[pats.split == "test"]
out = {}

print("=" * 100); print("A1. ANCHOR / FOLLOW-UP / DEATH STRUCTURE"); print("=" * 100)
for name, df in (("train", train), ("val", val), ("test", test)):
    q = df.anchor_year.quantile([0, .05, .25, .5, .75, .95, 1]).round(2).tolist()
    print(f"{name:5s} n={len(df):4d} anchor_year quantiles(0,5,25,50,75,95,100%)={q}")
print("max record end (train_val):", tv["end"].max().date(), " | died fraction train/val:", round(train.died.mean(), 3), round(val.died.mean(), 3))
print("death_in_window == died ? ", (tv.death_in_window == tv.died).mean())
print("death - end (days) quantiles for died:", (tv.loc[tv.died, "DEATHDATE"] - tv.loc[tv.died, "end"]).dt.days.quantile([0, .5, 1]).tolist())
end_max = tv["end"].max()
tv["alive_like"] = (end_max - tv["end"]).dt.days < 400
print("fraction train/val whose record ends within 400 days of the global max end:", tv.alive_like.mean().round(3),
      "| among those, died:", tv.loc[tv.alive_like, "died"].mean().round(3), "| among others, died:", tv.loc[~tv.alive_like, "died"].mean().round(3))
band = end_max.year - 5 - 1.1
print("fraction of TEST anchors in the alive-like recent band (anchor_year > max_end-5-1.1):", round((test.anchor_year > band).mean(), 3),
      " vs train/val:", round((tv.anchor_year > band).mean(), 3))
print("age at anchor quantiles train:", train.age.quantile([0, .25, .5, .75, 1]).round(1).tolist())

print(); print("per-target incidence among at-risk (train), split by died vs alive; RR = incidence(died)/incidence(alive)")
rows = []
Ytr, Ptr = Y.loc[train.PATIENT], P.loc[train.PATIENT]
for c in codes:
    atrisk = Ptr[c].values == 0
    y = Ytr[c].values[atrisk]; died = train.died.values[atrisk]
    inc = y.mean(); inc_d = y[died].mean() if died.any() else np.nan; inc_a = y[~died].mean()
    rows.append(dict(code=c, cond=short[c], n_atrisk=int(atrisk.sum()), pos=int(y.sum()), inc=round(inc, 4),
                     inc_died=round(inc_d, 4), inc_alive=round(inc_a, 4), RR=round(inc_d / max(inc_a, 1e-9), 2),
                     frac_pos_died=round(died[y == 1].mean(), 3)))
inc_df = pd.DataFrame(rows)
print(inc_df.to_string(index=False))
out["incidence_by_death"] = inc_df.to_dict("records")

print(); print("=" * 100); print("A2. VALIDATION mAP OF TRIVIAL SCORERS (prior->0 masking as in the project)"); print("=" * 100)
Yva, Pva = Y.loc[val.PATIENT], P.loc[val.PATIENT]

def hist_stats(df):
    pre = events[events.pre & events.PATIENT.isin(df.PATIENT)]
    g = pre.groupby("PATIENT")
    s = pd.DataFrame({"hist_years": g.years_before.max(), "log_events": np.log1p(g.size()),
                      "log_enc": np.log1p(g.kind.apply(lambda k: (k == "ENC").sum()))})
    return s.reindex(df.PATIENT).fillna(0)

for df in (train, val):
    hs = hist_stats(df)
    for col in hs.columns:
        pats.loc[df.index, col] = hs[col].values
train = pats[pats.split == "train"]; val = pats[pats.split == "val"]
train = train.assign(age2=train.age ** 2, ay2=train.anchor_year ** 2); val = val.assign(age2=val.age ** 2, ay2=val.anchor_year ** 2)

def feats(df, cols):
    return df[cols].astype(float).values

def eval_scorer(name, fit_predict):
    aps, aucs, per = [], [], {}
    for c in codes:
        tr_mask = Ptr[c].values == 0
        ytr = Ytr[c].values[tr_mask]
        p_val = fit_predict(c, tr_mask, ytr)
        p_val = np.where(Pva[c].values == 1, 0.0, p_val)
        yv = Yva[c].values
        ap = average_precision_score(yv, p_val) if yv.sum() else 0.0
        auc = roc_auc_score(yv, p_val) if 0 < yv.sum() < len(yv) else np.nan
        aps.append(ap); aucs.append(auc); per[c] = ap
    print(f"{name:70s} val mAP={np.mean(aps):.4f}  macroAUROC={np.nanmean(aucs):.4f}")
    out.setdefault("trivial_scorers", {})[name] = dict(mAP=float(np.mean(aps)), auroc=float(np.nanmean(aucs)), per=per)
    return per

def lr_factory(cols, C=1.0):
    def fp(c, tr_mask, ytr):
        if ytr.sum() < 2:
            return np.full(len(val), ytr.mean())
        sc = StandardScaler().fit(feats(train, cols)[tr_mask])
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(feats(train, cols)[tr_mask]), ytr)
        return m.predict_proba(sc.transform(feats(val, cols)))[:, 1]
    return fp

def gbm_factory(cols):
    def fp(c, tr_mask, ytr):
        if ytr.sum() < 2:
            return np.full(len(val), ytr.mean())
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=0)
        m.fit(feats(train, cols)[tr_mask], ytr)
        return m.predict_proba(feats(val, cols))[:, 1]
    return fp

per_const = eval_scorer("constant train incidence", lambda c, m, y: np.full(len(val), y.mean()))
per_agesex = eval_scorer("LR(age, age^2, sex)", lr_factory(["age", "age2", "female"]))
per_agesex_hist = eval_scorer("LR(age, age^2, sex, hist_years, log_events, log_enc)", lr_factory(["age", "age2", "female", "hist_years", "log_events", "log_enc"]))
per_anchor = eval_scorer("LR(+ anchor_year, anchor_year^2)  [calendar artifact]", lr_factory(["age", "age2", "female", "hist_years", "log_events", "log_enc", "anchor_year", "ay2"]))
per_gbm_noanchor = eval_scorer("GBM(age, sex, hist_years, log_events, log_enc)", gbm_factory(["age", "female", "hist_years", "log_events", "log_enc"]))
per_gbm_anchor = eval_scorer("GBM(+ anchor_year)  [calendar artifact]", gbm_factory(["age", "female", "hist_years", "log_events", "log_enc", "anchor_year"]))
per_oracle = eval_scorer("ORACLE LR(age, age^2, sex, died)  [not usable; ceiling of the artifact]", lr_factory(["age", "age2", "female", "died"]))

print(); print("per-condition AP: constant | age/sex LR | +anchor_year LR | GBM+anchor | oracle(died)")
cmp = pd.DataFrame({"cond": [short[c] for c in codes], "val_pos": Yva.sum().values, "const": list(per_const.values()),
                    "agesex": list(per_agesex.values()), "anchorLR": list(per_anchor.values()), "gbm_anchor": list(per_gbm_anchor.values()),
                    "oracle_died": list(per_oracle.values())}, index=codes).round(3)
print(cmp.to_string())

print(); print("=" * 100); print("A3. PRECURSOR LIFT (train, at-risk only): which pre-anchor codes predict each target"); print("=" * 100)
pre_tr = events[events.pre & (events.split == "train")].copy()
is_obs_cat = (pre_tr.kind == "OBS") & pre_tr.num.isna()
pre_tr["tok"] = pre_tr.kind + ":" + pre_tr.code
pre_tr.loc[is_obs_cat, "tok"] = "OBS:" + pre_tr.loc[is_obs_cat, "code"] + "|" + pre_tr.loc[is_obs_cat, "value"].fillna("").astype(str)
pre_tr.loc[pre_tr.kind == "ENC", "tok"] = "ENC:" + pre_tr.loc[pre_tr.kind == "ENC", "encclass"].astype(str)
pres = pre_tr[["PATIENT", "tok"]].drop_duplicates()
tok_counts = pres.tok.value_counts()
keep = tok_counts[(tok_counts >= 15) & (tok_counts <= len(train) - 15)].index
pres = pres[pres.tok.isin(keep)]
descmap = {}
for table, kind in (("conditions", "COND"), ("medications", "MED"), ("procedures", "PROC"), ("observations", "OBS"), ("careplans", "CARE"), ("immunizations", "IMM"), ("devices", "DEVICE"), ("allergies", "ALLERGY")):
    dfm = pd.read_csv(DATA / "train_val" / f"{table}.csv", usecols=["CODE", "DESCRIPTION"], dtype=str).drop_duplicates("CODE")
    descmap.update({f"{kind}:{r.CODE.strip()}": r.DESCRIPTION for r in dfm.itertuples()})
def tokdesc(t):
    base = t.split("|")[0]
    return (descmap.get(base, base)[:38] + ("|" + t.split("|", 1)[1][:16] if "|" in t else ""))
M = pd.crosstab(pres.PATIENT, pres.tok).reindex(train.PATIENT).fillna(0).astype(bool)
lift_rows = []
for c in codes:
    atrisk = Ptr[c].values == 0
    y = Ytr[c].values[atrisk].astype(bool); Mm = M.values[atrisk]
    base = y.mean()
    n_c = Mm.sum(0); pos_c = (Mm & y[:, None]).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = pos_c / n_c
    ok = (n_c >= 15) & (pos_c >= 5)
    order = np.argsort(-(np.nan_to_num(prec) * ok))[:5]
    print(f"{short[c]:36s} base={base:.3f} pos={int(y.sum())}:")
    for j in order:
        if not ok[j]:
            continue
        print(f"      {tokdesc(M.columns[j]):58s} P(y|c)={prec[j]:.2f} n={n_c[j]:4d} lift={prec[j]/base:5.1f}")
        lift_rows.append(dict(code=c, cond=short[c], token=M.columns[j], tokdesc=tokdesc(M.columns[j]), prec=float(prec[j]), n=int(n_c[j]), lift=float(prec[j] / base)))
out["precursor_lift"] = lift_rows

print(); print("=" * 100); print("A3b. NUMERIC OBSERVATIONS: AUROC of LAST pre-anchor value vs label (train at-risk)"); print("=" * 100)
num = pre_tr[(pre_tr.kind == "OBS") & pre_tr.num.notna()].sort_values("date")
last = num.groupby(["PATIENT", "code"]).num.last().unstack()
cnt = last.notna().sum(); last = last[cnt[cnt >= 150].index].reindex(train.PATIENT)
obsdesc = {k.split(":")[1]: v for k, v in descmap.items() if k.startswith("OBS:")}
num_rows = []
for c in codes:
    atrisk = Ptr[c].values == 0
    y = Ytr[c].values[atrisk]
    res = []
    for col in last.columns:
        x = last[col].values[atrisk]; m = ~np.isnan(x)
        if m.sum() < 100 or y[m].sum() < 5 or y[m].sum() == m.sum():
            continue
        auc = roc_auc_score(y[m], x[m])
        res.append((abs(auc - .5), auc, col, int(m.sum())))
    res.sort(reverse=True)
    txt = "; ".join(f"{obsdesc.get(col,'?')[:28]} AUC={auc:.2f} (n={n})" for _, auc, col, n in res[:3])
    print(f"{short[c]:36s} {txt}")
    num_rows.append(dict(code=c, top=[dict(code=col, desc=obsdesc.get(col, "?"), auc=float(auc), n=n) for _, auc, col, n in res[:3]]))
out["numeric_last_value_auc"] = num_rows

print(); print("=" * 100); print("A4. LABEL STRUCTURE: time-to-event and co-occurrence inside the 5y window (train+val)"); print("=" * 100)
Ft = d["first_targets"].loc[tv.PATIENT]
anc = tv.set_index("PATIENT").anchor
Ytv = Y.loc[tv.PATIENT]
print("years from anchor to first diagnosis for positives (median [p10,p90]):")
for c in codes:
    pos = Ytv[c].values == 1
    if pos.sum() == 0:
        continue
    t = ((Ft[c] - anc).dt.days / 365.25).values[pos]
    print(f"   {short[c]:36s} n={pos.sum():4d}  median={np.median(t):.2f}  p10={np.quantile(t,.1):.2f}  p90={np.quantile(t,.9):.2f}")
print(); print("label pairs with Jaccard >= 0.25 inside the window:")
Yv = Ytv.values.astype(bool)
for i in range(len(codes)):
    for j in range(i + 1, len(codes)):
        inter = (Yv[:, i] & Yv[:, j]).sum(); union = (Yv[:, i] | Yv[:, j]).sum()
        if union and inter / union >= 0.25:
            both = Yv[:, i] & Yv[:, j]
            dt = ((Ft[codes[j]] - Ft[codes[i]]).dt.days.values[both])
            print(f"   {short[codes[i]]:34s} & {short[codes[j]]:34s} J={inter/union:.2f} (n={inter}); median(day j - day i)={np.median(dt):.0f}")

print(); print("=" * 100); print("A5. PRIOR-STATUS: fraction of train patients already diagnosed before anchor (not at risk)"); print("=" * 100)
print(pd.Series({short[c]: round(Ptr[c].mean(), 3) for c in codes}).sort_values(ascending=False).head(12).to_string())

json.dump(out, open("dgp_results.json", "w"), indent=1, default=float)
print("saved dgp_results.json")
