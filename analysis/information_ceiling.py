"""Is the GBM already at the information ceiling of the data? Per target, compare the OOF AP of
(i) age/sex logistic, (ii) age/sex + top-10 precursor indicators (selected inside each fold), against the direct GBM OOF AP.
If (ii) ~ GBM, the remaining gap to 1.0 is irreducible randomness of the generator, not model capacity."""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore")
d = load(); pats, ev, Y, P, codes, targets = d["pats"], d["events"], d["Y"], d["P"], d["codes"], d["targets"]
desc = dict(zip(targets.CODE, targets.DESCRIPTION)); short = {c: desc[c][:32] for c in codes}
tr_ids = pats[pats.split == "train"].PATIENT.values
pre = ev[ev.pre & ev.PATIENT.isin(tr_ids)].copy()
is_cat = (pre.kind == "OBS") & pre.num.isna(); is_num = (pre.kind == "OBS") & pre.num.notna()
pre["tok"] = pre.kind + ":" + pre.code; pre.loc[is_cat, "tok"] = "OBS:" + pre.loc[is_cat, "code"] + "|" + pre.loc[is_cat, "value"].fillna("").astype(str)
pre = pre[~is_num]
M = pd.crosstab(pre.PATIENT, pre.tok).reindex(tr_ids).fillna(0).gt(0).values.astype(np.float32); toks = pd.crosstab(pre.PATIENT, pre.tok).columns
Ytr, Ptr = Y.loc[tr_ids].values, P.loc[tr_ids].values.astype(bool)
age = pats.set_index("PATIENT").age.reindex(tr_ids).values; fem = pats.set_index("PATIENT").female.reindex(tr_ids).values
S = np.c_[(age - 60) / 20, ((age - 60) / 20) ** 2, fem]
folds = list(KFold(5, shuffle=True, random_state=0).split(tr_ids))
ref = pd.read_csv(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\analysis\v2\per_condition_v2.csv", index_col=0); ref.index = ref.index.astype(str)
rows = []
for j, c in enumerate(codes):
    atrisk = ~Ptr[:, j]; y = Ytr[:, j]
    oof_s = np.zeros(len(tr_ids)); oof_t = np.zeros(len(tr_ids))
    for a, b in folds:
        a_ = a[atrisk[a]]
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(S[a_], y[a_]); oof_s[b] = lr.predict_proba(S[b])[:, 1]
        # top-10 precursor tokens by precision within the fold's training part (support >= 15, >= 5 positives)
        Ma = M[a_]; n = Ma.sum(0); pos = (Ma * y[a_, None]).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"): prec = np.where((n >= 15) & (pos >= 5), pos / n, 0)
        top = np.argsort(-prec)[:10]
        Xa = np.c_[S[a_], Ma[:, top]]; Xb = np.c_[S[b], M[b][:, top]]
        lr2 = LogisticRegression(C=1.0, max_iter=3000).fit(Xa, y[a_]); oof_t[b] = lr2.predict_proba(Xb)[:, 1]
    oof_s[~atrisk] = 0; oof_t[~atrisk] = 0
    rows.append(dict(cond=short[c], train_pos=int(y[atrisk].sum()), ap_agesex=average_precision_score(y, oof_s), ap_agesex_top10=average_precision_score(y, oof_t), ap_gbm=float(ref.loc[c, "oof_gbm"])))
df = pd.DataFrame(rows).round(3); df["gbm_minus_lr10"] = (df.ap_gbm - df.ap_agesex_top10).round(3)
pd.set_option("display.width", 200); print(df.to_string(index=False))
print("\nmeans:", df[["ap_agesex", "ap_agesex_top10", "ap_gbm"]].mean().round(4).to_dict())
df.to_csv(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\analysis\information_ceiling.csv", index=False)
