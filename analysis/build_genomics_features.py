"""Per-patient features from the optional modalities (task README: 'You may use any modality provided').
genomics/<Id>_dna.csv : 161 rsIDs x 2 strands; VARIANT=true marks a variant allele on that strand.
imaging/<Id>.dcm      : head MR; only counted when its StudyDate is on/before the patient's anchor (pre-anchor event rule).
Output: analysis/clean/genomics_features.parquet (index PATIENT; has_dna, gen_* (NaN when no file), has_dicom_pre)."""
import glob, os, time, numpy as np, pandas as pd
from pathlib import Path
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "research_intern_coding_test"; OUT = ROOT / "analysis" / "clean"
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
# anchors: train/val = last encounter (STOP, else START) - 5y ; test = test_anchors.csv
enc = pd.read_csv(DATA / "train_val" / "encounters.csv", dtype=str, usecols=["PATIENT", "START", "STOP"])
end = enc.STOP.where(enc.STOP.notna() & (enc.STOP != ""), enc.START); end = pd.to_datetime(end, utc=True, errors="coerce").dt.tz_localize(None)
last = end.groupby(enc.PATIENT).max(); anchor = (last - pd.DateOffset(years=5)).rename("anchor")
ta = pd.read_csv(DATA / "test_anchors.csv", dtype=str); ta = pd.Series(pd.to_datetime(ta.iloc[:, 1], utc=True, errors="coerce").dt.tz_localize(None).values, index=ta.iloc[:, 0].values, name="anchor")
anchor = pd.concat([anchor, ta]); anchor = anchor[~anchor.index.duplicated()]
# genomics
rows = []
for f in glob.glob(str(DATA / "genomics" / "*_dna.csv")):
    pid = os.path.basename(f).split("_dna")[0]; g = pd.read_csv(f, dtype=str)
    g["var"] = g.VARIANT.str.strip().str.lower() == "true"; g["GENE"] = g.GENE.str.strip('"').str.strip()
    rs = g.groupby("INDEX_PREFIX")["var"].sum(); gene = g[g["var"]].groupby("GENE").size()
    r = {"PATIENT": pid, "gen_n_var_alleles": int(g["var"].sum()), "gen_n_var_rsids": int((rs > 0).sum()),
         "gen_n_pathogenic": int((g["var"] & g.CLINICAL_SIGNIFICANCE.str.contains("athogenic", na=False)).sum()),
         "gen_n_uncertain": int((g["var"] & g.CLINICAL_SIGNIFICANCE.str.contains("ncertain", na=False)).sum())}
    r.update({f"gen_rs_{k}": int(v) for k, v in rs.items()}); r.update({f"gen_gene_{k}": int(v) for k, v in gene.items()})
    rows.append(r)
G = pd.DataFrame(rows).set_index("PATIENT")
gene_cols = [c for c in G.columns if c.startswith("gen_gene_")]; G[gene_cols] = G[gene_cols].fillna(0)
# dicom (pre-anchor only)
import pydicom
drows = []
for f in glob.glob(str(DATA / "imaging" / "*.dcm")):
    pid = os.path.basename(f)[:-4]
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True); sd = str(getattr(ds, "StudyDate", "") or getattr(ds, "SeriesDate", "") or getattr(ds, "ContentDate", ""))
        d = pd.to_datetime(sd, format="%Y%m%d", errors="coerce")
    except Exception as e:
        d = pd.NaT
    drows.append({"PATIENT": pid, "dicom_date": d})
D = pd.DataFrame(drows).set_index("PATIENT"); D["anchor"] = anchor.reindex(D.index)
D["has_dicom_pre"] = ((D.dicom_date.notna()) & (D.anchor.notna()) & (D.dicom_date <= D.anchor)).astype(int)
print("dicom: files", len(D), "| with date", int(D.dicom_date.notna().sum()), "| dated on/before anchor", int(D.has_dicom_pre.sum()), "| split of all dicom holders:", spl.set_index("PATIENT").split.reindex(D.index).value_counts().to_dict())
F = pd.DataFrame(index=spl.PATIENT.values); F.index.name = "PATIENT"
F["has_dna"] = F.index.isin(G.index).astype(int)
F = F.join(G, how="left")
F["has_dicom_pre"] = D.has_dicom_pre.reindex(F.index).fillna(0).astype(int)
F.to_parquet(OUT / "genomics_features.parquet")
print(f"saved {F.shape} -> {OUT/'genomics_features.parquet'}  has_dna by split:", F.groupby(spl.set_index('PATIENT').split.reindex(F.index)).has_dna.sum().to_dict(), f" [{time.time()-t0:.0f}s]")
print("gen cols:", len([c for c in F.columns if c.startswith('gen_')]), "| rs cols:", len([c for c in F.columns if c.startswith('gen_rs_')]), "| gene cols:", len(gene_cols))
