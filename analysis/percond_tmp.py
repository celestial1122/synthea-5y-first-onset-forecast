import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from ehr_load import load
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
ids = {s: pats[pats.split == s].PATIENT.values for s in ("train","val","test")}
Ytr = Y.loc[ids["train"]].values; Ptr = P.loc[ids["train"]].values.astype(bool)
Yva = Y.loc[ids["val"]].values; Pva = P.loc[ids["val"]].values.astype(bool)
TERMINAL = ["88805009","22298006","233604007","126906006","92691004","162573006","424132000","254637007","314994000","713197008"]
ACCIDENT = ["444814009","195662009","10509002","44465007","43878008","36971009","75498004","39848009","263102004","62106007","58150001","16114001","70704007","283371005","284551006"]
grp = ["term" if c in TERMINAL else "acc" if c in ACCIDENT else "chain" for c in codes]
best = {}
for name, path in [("lgb_clean","clean/oof.npy"),("lgb_sweep_avg","lgb_sweep/avg_oof.npy"),("pp_yearly","pp2/pp_yearly_uniform_oof.npy"),("pp_lgb","pp_lgb/person-period_uniform_oof.npy")]:
    o = np.load(path); best[name] = np.array([average_precision_score(Ytr[:,j], np.where(Ptr[:,j],0,o[:,j])) for j in range(40)])
fv = pd.read_csv("final/val_predictions_final.csv", dtype={"patient_id":str}).set_index("patient_id").reindex(ids["val"])[codes].values
final_val = np.array([average_precision_score(Yva[:,j], fv[:,j]) for j in range(40)])
pos = np.array([int(Ytr[~Ptr[:,j], j].sum()) for j in range(40)]); atrisk = np.array([int((~Ptr[:,j]).sum()) for j in range(40)])
df = pd.DataFrame({"cond":[desc[c][:34] for c in codes], "grp":grp, "pos":pos, "prev":(pos/atrisk).round(3), **{k:v.round(3) for k,v in best.items()}, "final_val":final_val.round(3)}, index=codes)
df["best_oof"] = df[list(best)].max(1); df["lift"] = (df.best_oof/df.prev).round(2)
pd.set_option("display.width", 250); print(df.sort_values(["grp","lift"]).to_string())
print(df.groupby("grp")[list(best)+["final_val","prev"]].mean().round(4))
print("pos death in window (train):", pats[pats.split=="train"].died.mean())
