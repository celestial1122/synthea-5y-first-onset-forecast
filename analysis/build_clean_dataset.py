"""Build the cleaned, leak-audited patient-level dataset (our representation of the raw tables).

Rules (see docs/data_audit.md):
  * inputs: the 10 structured tables incl. imaging_studies; events strictly before the anchor (calendar date)
  * genomics / DICOM side files are NOT used (generated coherently with diagnoses -> target leakage)
  * STOP dates later than the anchor are blanked in train/val to mimic the test truncation; "active at anchor" = STOP blank
  * REASONCODE columns are not used (redundant with conditions; 2-3% reference later-dated diagnoses)
  * same-day duplicate rows (patient, date, table, code, value) are collapsed
  * numeric observations keyed by code (unit checked in the audit); raw last/min/max/trend, no quantile binning
  * optional --no-qol drops the QALY/DALY/QOLS bookkeeping observations (sensitivity analysis)
Outputs analysis/clean/{X.parquet, manifest.json}; row order = ehr_load pats order; column groups documented in manifest.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from ehr_load import load, DATA, _d
t0 = time.time()
NO_QOL = "--no-qol" in sys.argv
OUT = Path(__file__).parent / "clean"; OUT.mkdir(exist_ok=True)
d = load(); pats, events, codes = d["pats"], d["events"], d["codes"]
pidx = pats.set_index("PATIENT"); ids = pats.PATIENT.values
train_ids = set(pats[pats.split == "train"].PATIENT)
MIN_SUPPORT_TOK, MIN_SUPPORT_NUM = 20, 50
manifest = {"rules": __doc__, "no_qol": NO_QOL, "n_patients": int(len(ids)), "excluded_sources": ["genomics/*_dna.csv (coherent with stroke cohort)", "imaging/*.dcm (coherent with Alzheimer cohort)", "REASONCODE/REASONDESCRIPTION", "STOP as a date feature", "DEATHDATE", "costs/payer/identity columns"]}

# ---- events: cache tables + imaging_studies
ev = events[["PATIENT", "date", "kind", "code", "value", "units", "num", "encclass", "split", "anchor", "years_before", "pre"]].copy()
im = pd.concat([pd.read_csv(DATA / f / "imaging_studies.csv", dtype=str) for f in ("train_val", "test")], ignore_index=True)
im = pd.DataFrame({"PATIENT": im.PATIENT, "date": _d(im.DATE), "kind": "IMG", "code": im.MODALITY_CODE.str.strip() + "|" + im.BODYSITE_CODE.str.strip(), "value": pd.NA, "units": "", "num": np.nan, "encclass": pd.NA})
im = im.merge(pats[["PATIENT", "split", "anchor"]], on="PATIENT"); im["years_before"] = (im.anchor - im.date).dt.days / 365.25; im["pre"] = im.date < im.anchor
ev = pd.concat([ev, im], ignore_index=True)
n_before = len(ev)
ev = ev[ev.pre].drop_duplicates(["PATIENT", "date", "kind", "code", "value"])
manifest["events_pre_anchor_after_dedup"] = int(len(ev)); manifest["rows_total_including_post_anchor"] = int(n_before)
if NO_QOL:
    ev = ev[~((ev.kind == "OBS") & ev.code.isin(["QALY", "DALY", "QOLS"]))]
is_num = (ev.kind == "OBS") & ev.num.notna(); is_cat = (ev.kind == "OBS") & ev.num.isna()
ev["tok"] = ev.kind + ":" + ev.code
ev.loc[is_cat, "tok"] = "OBS:" + ev.loc[is_cat, "code"] + "|" + ev.loc[is_cat, "value"].fillna("").astype(str)
ev.loc[ev.kind == "ENC", "tok"] = "ENC:" + ev.loc[ev.kind == "ENC", "encclass"].astype(str)

# ---- token features (vocabulary fitted on train only)
sup = ev[~is_num & ev.PATIENT.isin(train_ids)][["PATIENT", "tok"]].drop_duplicates().tok.value_counts(); KEEP = sup[sup >= MIN_SUPPORT_TOK].index
sub = ev[ev.tok.isin(KEEP) & ~is_num]; g = sub.groupby(["PATIENT", "tok"])
blocks = {
    "cnt": g.size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0),
    "c5": sub[sub.years_before <= 5].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0),
    "c1": sub[sub.years_before <= 1].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0),
    "rec": g.years_before.min().unstack().reindex(index=ids, columns=KEEP),
    "first": g.years_before.max().unstack().reindex(index=ids, columns=KEEP),
}
# ---- numeric observations (train-supported codes), raw statistics
# numeric codes are keyed by code+unit: 5 codes carry two units with different scales (e.g. eGFR mL/min vs mL/min/1.73m2)
ev.loc[is_num, "code"] = ev.loc[is_num, "code"] + "|" + ev.loc[is_num, "units"].fillna("").astype(str)
nsup = ev[is_num & ev.PATIENT.isin(train_ids)].groupby("code").PATIENT.nunique(); NKEEP = nsup[nsup >= MIN_SUPPORT_NUM].index
num = ev[is_num & ev.code.isin(NKEEP)].sort_values("date"); gn = num.groupby(["PATIENT", "code"]).num
blocks.update({
    "last": gn.last().unstack().reindex(index=ids, columns=NKEEP), "max": gn.max().unstack().reindex(index=ids, columns=NKEEP), "min": gn.min().unstack().reindex(index=ids, columns=NKEEP),
    "m5": num[num.years_before <= 5].groupby(["PATIENT", "code"]).num.mean().unstack().reindex(index=ids, columns=NKEEP),
    "std": gn.std().unstack().reindex(index=ids, columns=NKEEP), "nobs": gn.size().unstack().reindex(index=ids, columns=NKEEP).fillna(0),
    "trend": (gn.last() - num[num.years_before > 2].groupby(["PATIENT", "code"]).num.mean()).unstack().reindex(index=ids, columns=NKEEP),
})
# ---- medications active at anchor (STOP blanked when > anchor, as in the test export)
med = pd.concat([pd.read_csv(DATA / f / "medications.csv", dtype=str, usecols=["START", "STOP", "PATIENT", "CODE"]) for f in ("train_val", "test")], ignore_index=True)
med["start"] = _d(med.START); med["stop"] = _d(med.STOP); med["anchor"] = med.PATIENT.map(pidx.anchor); med = med[med.start < med.anchor]
med.loc[med.stop > med.anchor, "stop"] = pd.NaT                      # mimic test truncation
med["active"] = med.stop.isna(); act = med[med.active]
asup = act[act.PATIENT.isin(train_ids)].groupby("CODE").PATIENT.nunique(); AKEEP = asup[asup >= MIN_SUPPORT_TOK].index
blocks["active"] = act[act.CODE.isin(AKEEP)].groupby(["PATIENT", "CODE"]).size().unstack(fill_value=0).gt(0).astype(float).reindex(index=ids, columns=AKEEP).fillna(0)
# ---- static / summary
st = pd.DataFrame(index=ids); st["age"] = pidx.age.reindex(ids); st["female"] = pidx.female.reindex(ids)
for r in ("white", "black", "asian", "hispanic"): st[f"race_{r}"] = (pidx.RACE.reindex(ids) == r).astype(int)
gp = ev.groupby("PATIENT"); st["hist_years"] = gp.years_before.max().reindex(ids).fillna(0); st["n_events"] = gp.size().reindex(ids).fillna(0)
st["n_events_5y"] = ev[ev.years_before <= 5].groupby("PATIENT").size().reindex(ids).fillna(0)
for k in ("COND", "MED", "PROC", "IMG"): st[f"n_{k.lower()}_codes"] = ev[ev.kind == k].groupby("PATIENT").code.nunique().reindex(ids).fillna(0)
st["n_tokens"] = gp.tok.nunique().reindex(ids).fillna(0); st["n_active_meds"] = act.groupby("PATIENT").CODE.nunique().reindex(ids).fillna(0)
enc = ev[ev.kind == "ENC"]; rates = enc.groupby(["PATIENT", "encclass"]).size().unstack(fill_value=0).reindex(index=ids).fillna(0).div(st.hist_years.clip(lower=1.0), axis=0)
blocks["encrate"] = rates; blocks["static"] = st; blocks["prior"] = d["P"].reindex(ids).astype(float)
X = pd.concat([v.add_prefix(k + "_") if k not in ("static",) else v for k, v in blocks.items()], axis=1).astype(np.float32)
X.index.name = "PATIENT"
X.to_parquet(OUT / ("X_noqol.parquet" if NO_QOL else "X.parquet"))
manifest.update({"n_features": int(X.shape[1]), "blocks": {k: int(v.shape[1]) for k, v in blocks.items()}, "token_vocab": int(len(KEEP)), "numeric_codes": int(len(NKEEP)), "active_med_codes": int(len(AKEEP)),
                 "min_support_token_patients": MIN_SUPPORT_TOK, "min_support_numeric_patients": MIN_SUPPORT_NUM, "seconds": round(time.time() - t0, 1)})
json.dump(manifest, open(OUT / ("manifest_noqol.json" if NO_QOL else "manifest.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in manifest.items() if k != "rules"}, indent=1))
