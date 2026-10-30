"""Mechanism probe: for each non-accident target, how much AP do a few 'generator-mechanism' features give on their own
(univariate ranking on the at-risk official training rows, so no fitting optimism), and does a tiny GBM restricted to a
target-specific feature subset beat the full-feature GBM OOF? If yes, feature dilution (2300 columns vs 30-200 positives)
is a real problem and target-specific subsets / monotone constraints are worth building."""
import json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from ehr_load import load
warnings.filterwarnings("ignore"); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"
d = load(); pats, events, Y, P, codes, targets = d["pats"], d["events"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
X = pd.read_parquet(HERE / "clean" / "X.parquet").reindex(pats.PATIENT); tr = (pats.split == "train").values
Xtr = X[tr]; Ytr = Y.values[tr]; Ptr = P.values[tr].astype(bool)
ref = pd.read_csv(HERE / "lgb_sweep" / "per_condition_lgb_sweep.csv", index_col=0); ref.index = ref.index.astype(str)
# numeric code descriptions
obs = events[(events.kind == "OBS") & events.num.notna()][["code", "units"]].drop_duplicates()
odesc = pd.read_csv(ROOT / "research_intern_coding_test" / "train_val" / "observations.csv", usecols=["CODE", "DESCRIPTION"], dtype=str).drop_duplicates("CODE").set_index("CODE").DESCRIPTION if (ROOT / "research_intern_coding_test" / "train_val" / "observations.csv").exists() else None
numcodes = sorted({c.split("last_", 1)[1] for c in X.columns if c.startswith("last_")})
print("numeric codes:", len(numcodes))
if odesc is not None:
    for c in numcodes: print(f"  {c:12s} {odesc.get(c.split('|')[0], '?')}")
def ap(j, s):
    m = ~Ptr[:, j]; return average_precision_score(Ytr[m, j], s[m]) if Ytr[m, j].sum() else np.nan
def col(name): return Xtr[name].fillna(Xtr[name].median() if Xtr[name].notna().any() else 0).values if name in Xtr else np.zeros(len(Xtr))
def has(prefix): return [c for c in Xtr.columns if c.startswith(prefix)]
bmi = [c for c in numcodes if c.startswith("39156-5")]; bmi = bmi[0] if bmi else None
HYP = {}
if bmi:
    HYP["162864005"] = {"last_bmi": col(f"last_{bmi}"), "max_bmi": col(f"max_{bmi}"), "proj_bmi_2.5y": col(f"last_{bmi}") + 2.5 * col(f"trend_{bmi}") / 3.0, "bmi_minus_age_penalty": col(f"last_{bmi}") - 0.0 * col("age")}
for c in numcodes:
    pass
a1c = [c for c in numcodes if c.startswith("4548-4")]; glu = [c for c in numcodes if c.startswith("2339-0")]
if a1c: HYP["15777000"] = {"last_a1c": col(f"last_{a1c[0]}"), "max_a1c": col(f"max_{a1c[0]}"), "last_bmi": col(f"last_{bmi}") if bmi else np.zeros(len(Xtr)), "age": col("age")}
HYP["72892002"] = {"age_band": -np.abs(col("age") - 27) * col("female"), "female": col("female"), "-age": -col("age")}
HYP["26929004"] = {"age": col("age")}
HYP["64859006"] = {"age*female": col("age") * col("female"), "age": col("age")}
HYP["271737000"] = {"age": col("age"), "n_cond": col("n_cond_codes"), "encrate_ambulatory": col("encrate_ambulatory")}
sbp = [c for c in numcodes if c.startswith("8480-6")]; chol = [c for c in numcodes if c.startswith("2093-3")]; ldl = [c for c in numcodes if c.startswith("18262-6")]; hdl = [c for c in numcodes if c.startswith("2085-9")]
cv = {"age": col("age"), "male": 1 - col("female")}
if sbp: cv["last_sbp"] = col(f"last_{sbp[0]}"); cv["max_sbp"] = col(f"max_{sbp[0]}")
if chol: cv["last_chol"] = col(f"last_{chol[0]}")
if ldl: cv["last_ldl"] = col(f"last_{ldl[0]}")
if hdl: cv["-last_hdl"] = -col(f"last_{hdl[0]}")
for c in ("53741008", "230690007", "49436004", "22298006", "88805009"): HYP[c] = dict(cv)
print("\nunivariate AP on at-risk official train rows (vs full GBM OOF, prevalence):")
for c, feats in HYP.items():
    j = codes.index(c); m = ~Ptr[:, j]; prev = Ytr[m, j].mean()
    s = ", ".join(f"{k}={ap(j, v):.3f}" for k, v in feats.items())
    print(f"  {desc[c][:34]:34s} prev={prev:.3f} GBM={ref.loc[c, 'nl4_mcs15'] if c in ref.index else float('nan'):.3f} | {s}")
# tiny GBM on target-specific subsets (5-fold OOF x 2) vs full-feature GBM
print("\nsubset-GBM OOF (depth 3, 300 trees) vs full GBM:")
SUBSETS = {"162864005": has("last_39156") + has("max_39156") + has("min_39156") + has("trend_39156") + has("m5_39156") + has("nobs_39156") + ["age", "female"],
           "15777000": has("last_4548") + has("max_4548") + has("last_2339") + has("max_2339") + has("last_39156") + has("trend_39156") + ["age", "female"],
           "72892002": ["age", "female"] + [c for c in Xtr.columns if c.startswith("cnt_MED:") or c.startswith("active_")][:200] + has("prior_72892002"),
           "26929004": ["age", "female"] + has("prior_") + has("encrate_"),
           "64859006": ["age", "female"] + has("prior_") + [c for c in Xtr.columns if "COND:" in c and c.startswith("cnt_")],
           "53741008": ["age", "female"] + has("last_8480") + has("last_8462") + has("last_2093") + has("last_18262") + has("last_2085") + has("max_8480") + has("prior_") + [c for c in Xtr.columns if "72166-2" in c] + has("last_39156"),
           "230690007": ["age", "female"] + has("last_8480") + has("last_8462") + has("last_2093") + has("last_18262") + has("prior_") + [c for c in Xtr.columns if "72166-2" in c],
           "271737000": ["age", "female"] + has("prior_") + has("encrate_") + has("last_") }
PP = np.load(HERE / "pp2" / "pp_yearly_uniform_oof.npy")
cfg = dict(objective="binary", verbose=-1, n_jobs=8, learning_rate=0.03, n_estimators=300, num_leaves=8, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0)
for c, sub in SUBSETS.items():
    sub = [s for s in dict.fromkeys(sub) if s in Xtr.columns]; j = codes.index(c); m = ~Ptr[:, j]; Xs = Xtr[sub].values; res = []
    for r in range(2):
        oof = np.zeros(len(Xtr))
        for a, b in KFold(5, shuffle=True, random_state=r).split(Xs):
            mm = m[a]; mdl = lgb.LGBMClassifier(random_state=r, **cfg).fit(Xs[a][mm], Ytr[a, j][mm]); oof[b] = mdl.predict_proba(Xs[b])[:, 1]
        res.append(average_precision_score(Ytr[m, j], oof[m]))
    print(f"  {desc[c][:34]:34s} subset({len(sub):4d} feats) OOF={np.mean(res):.3f}  full GBM={ref.loc[c, 'nl4_mcs15']:.3f}  pp_yearly={average_precision_score(Ytr[m, j], PP[m, j]):.3f}  [{time.time()-t0:.0f}s]")
