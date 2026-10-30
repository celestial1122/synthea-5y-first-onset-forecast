"""Comprehensive data audit, part B: side files (genomics, DICOM), REASONCODE columns, and the full inventory of
future-information channels with quantified strength."""
import warnings, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from ehr_load import load, DATA, _d
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
d = load(); pats, ev, Y, P, codes, targets = d["pats"], d["events"], d["Y"], d["P"], d["codes"], d["targets"]
pidx = pats.set_index("PATIENT"); desc = dict(zip(targets.CODE, targets.DESCRIPTION)); short = {c: desc[c][:30] for c in codes}
tv = pats[pats.split != "test"].PATIENT.values
cond_all = pd.read_csv(DATA / "train_val" / "conditions.csv", dtype=str); cond_all["date"] = _d(cond_all.START)
cd = cond_all.drop_duplicates("CODE").set_index("CODE").DESCRIPTION

print("#" * 100); print("B1. GENOMICS: who got a DNA file? (full-record condition enrichment, train+val)"); print("#" * 100)
gid = [os.path.basename(f).split("_dna")[0] for f in glob.glob(str(DATA / "genomics" / "*_dna.csv"))]
has = pd.Series(pats.PATIENT.isin(gid).values, index=pats.PATIENT)
ever = cond_all.groupby(["PATIENT", "CODE"]).size().unstack(fill_value=0).gt(0).reindex(tv).fillna(False)
h = has.reindex(tv).values
enr = pd.DataFrame({"with_dna": ever[h].mean(), "without_dna": ever[~h].mean(), "n_with": ever[h].sum()})
enr["ratio"] = enr.with_dna / enr.without_dna.clip(lower=1e-3); enr = enr[enr.n_with >= 30].sort_values("ratio", ascending=False)
enr.insert(0, "desc", [str(cd.get(c, ""))[:40] for c in enr.index]); print(enr.head(15).round(3).to_string())
print("\nfraction of DNA patients (train+val) that EVER have stroke:", round(ever.loc[h, "230690007"].mean(), 3), "| MI:", round(ever.loc[h, "22298006"].mean(), 3), "| either:", round((ever.loc[h, "230690007"] | ever.loc[h, "22298006"]).mean(), 3))
print("fraction of all train+val stroke-ever patients that have DNA:", round(has.reindex(tv)[ever["230690007"].values].mean(), 3), "| of MI-ever:", round(has.reindex(tv)[ever["22298006"].values].mean(), 3))
# timing: for DNA patients with stroke, when is the stroke relative to the anchor?
fs = cond_all[cond_all.CODE == "230690007"].groupby("PATIENT").date.min()
rel = ((fs - pidx.anchor.reindex(fs.index)).dt.days / 365.25)
print("stroke timing relative to anchor among DNA patients: before anchor", round((rel[has.reindex(rel.index).fillna(False).values] < 0).mean(), 3), "| in window:", round(((rel >= 0) & (rel <= 5))[has.reindex(rel.index).fillna(False).values].mean(), 3))
# variant determinism
G = pd.read_pickle("genomics_long.pkl"); G["GENE"] = G.GENE.str.strip('"'); G["var"] = G.VARIANT.str.lower() == "true"
car = G.groupby(["PATIENT", "INDEX_PREFIX"])["var"].max().unstack(fill_value=False)
ids = [p for p in car.index if p in set(tv)]
stroke_ever = ever.loc[ids, "230690007"].values; mi_ever = ever.loc[ids, "22298006"].values
rows = []
for rs in car.columns:
    x = car.loc[ids, rs].values
    if x.sum() < 20: continue
    rows.append((rs, G.loc[G.INDEX_PREFIX == rs, "GENE"].iloc[0], int(x.sum()), round(stroke_ever[x].mean(), 3), round(stroke_ever[~x].mean(), 3), round(mi_ever[x].mean(), 3), round(mi_ever[~x].mean(), 3)))
vt = pd.DataFrame(rows, columns=["rsid", "gene", "carriers", "stroke_ever|carrier", "stroke_ever|non", "MI_ever|carrier", "MI_ever|non"]).sort_values("stroke_ever|carrier", ascending=False)
print("\nvariant carriers vs EVER-stroke / EVER-MI among DNA patients (train+val):"); print(vt.head(12).to_string(index=False)); print(vt.tail(5).to_string(index=False))
n_var = car.loc[ids].sum(1); print("number of variant alleles carried: stroke-ever patients median", int(n_var[stroke_ever].median()), "| others", int(n_var[~stroke_ever].median()))
print("test patients with DNA:", int(has[pats[pats.split == 'test'].PATIENT].sum()), "| val:", int(has[pats[pats.split == 'val'].PATIENT].sum()))

print("\n" + "#" * 100); print("B2. DICOM IMAGING"); print("#" * 100)
dcm = [os.path.basename(f)[:-4] for f in glob.glob(str(DATA / "imaging" / "*.dcm"))]; hd = pd.Series(pats.PATIENT.isin(dcm).values, index=pats.PATIENT)
print("dcm patients by split:", pats[hd.values].split.value_counts().to_dict())
for c in ["26929004", "22325002", "723857007", "230690007"]:
    print(f"   EVER {short[c]:30s} among dcm patients: {ever.loc[hd.reindex(tv).values, c].mean():.3f}  vs others {ever.loc[~hd.reindex(tv).values, c].mean():.3f}")
im = pd.concat([pd.read_csv(DATA / f / "imaging_studies.csv", dtype=str) for f in ("train_val", "test")]); im["date"] = _d(im.DATE); im["anchor"] = im.PATIENT.map(pidx.anchor); im["pre"] = im.date < im.anchor
print("imaging_studies: test rows", int((im.PATIENT.map(pidx.split) == "test").sum()), "| test rows post-anchor:", int(((im.PATIENT.map(pidx.split) == "test") & ~im.pre).sum()), "(should be 0)")

print("\n" + "#" * 100); print("B3. REASONCODE columns: do reasons ever reference a condition first diagnosed AFTER the row's own date?"); print("#" * 100)
first = cond_all.groupby(["PATIENT", "CODE"]).date.min()
for table, dcol in (("medications", "START"), ("careplans", "START"), ("procedures", "DATE"), ("encounters", "START")):
    df = pd.read_csv(DATA / "train_val" / f"{table}.csv", dtype=str, usecols=[dcol, "PATIENT", "REASONCODE"]); df = df[df.REASONCODE.notna()]
    df["date"] = _d(df[dcol]); key = list(zip(df.PATIENT, df.REASONCODE.str.strip()))
    fd = first.reindex(pd.MultiIndex.from_tuples(key)).values
    known = ~pd.isna(fd); later = (pd.to_datetime(fd[known]) > df.date.values[known])
    tgt = df.REASONCODE.str.strip().isin(codes)
    print(f"  {table:12s} rows with reason={len(df):6d} | reason is a known condition code: {known.mean():.3f} | reason first-diagnosed AFTER row date: {later.mean():.4f} | reason is a TARGET code: {tgt.mean():.3f}")
# reason codes that are target conditions and appear pre-anchor while the condition itself is first diagnosed in the window (leak if used)
df = pd.read_csv(DATA / "train_val" / "medications.csv", dtype=str, usecols=["START", "PATIENT", "REASONCODE"]); df = df[df.REASONCODE.notna()]; df["date"] = _d(df.START)
df["anchor"] = df.PATIENT.map(pidx.anchor); df = df[df.date < df.anchor]; df["rc"] = df.REASONCODE.str.strip()
fd = first.reindex(pd.MultiIndex.from_tuples(list(zip(df.PATIENT, df.rc)))).values
leak = (pd.to_datetime(fd) >= df.anchor.values) & df.rc.isin(codes).values
print("  pre-anchor medication rows whose REASON is a target condition first diagnosed in the FUTURE window:", int(np.nansum(leak)))

print("\n" + "#" * 100); print("B4. FUTURE-INFORMATION CHANNEL INVENTORY (strength = AUROC for death-in-window or target, train+val)"); print("#" * 100)
died = pidx.died.reindex(tv).values.astype(int)
print(f"  anchor calendar year alone -> death AUROC = {roc_auc_score(died, -pidx.anchor_year.reindex(tv).values):.3f}")
qol = ev[(ev.kind == 'OBS') & ev.code.isin(['QALY', 'DALY', 'QOLS']) & ev.pre]; hq = pd.Series(tv, index=tv).isin(qol.PATIENT.unique()).values.astype(int)
print(f"  has any pre-anchor QALY/DALY/QOLS observation -> death AUROC = {roc_auc_score(died, -hq):.3f}")
print(f"  has DNA file -> death AUROC = {roc_auc_score(died, has.reindex(tv).values.astype(int)):.3f} ; -> stroke-in-window AUROC (at risk) = {roc_auc_score(Y.loc[tv, '230690007'].values[P.loc[tv, '230690007'].values == 0], has.reindex(tv).values.astype(int)[P.loc[tv, '230690007'].values == 0]):.3f}")
print(f"  has DICOM -> abnormal-gait-in-window AUROC (at risk) = {roc_auc_score(Y.loc[tv, '22325002'].values[P.loc[tv, '22325002'].values == 0], hd.reindex(tv).values.astype(int)[P.loc[tv, '22325002'].values == 0]):.3f} (only 2 test patients have DICOM)")
