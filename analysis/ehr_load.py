"""Shared loader: anchors, labels, long event table. Mirrors the project's conventions."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "research_intern_coding_test"
CACHE = Path(__file__).parent / "ehr_cache.pkl"


def _d(s):
    return pd.to_datetime(s.astype("string").str.slice(0, 10), format="%Y-%m-%d", errors="coerce")


def _shift5(ts):
    return ts - pd.DateOffset(years=5)


def load(force=False):
    if CACHE.exists() and not force:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    splits = pd.read_csv(DATA / "patient_splits.csv").rename(columns={"Id": "PATIENT"})
    targets = pd.read_csv(DATA / "target_conditions.csv", dtype=str)
    codes = targets.CODE.tolist()
    pats = pd.concat([pd.read_csv(DATA / f / "patients.csv", dtype=str) for f in ("train_val", "test")], ignore_index=True)
    pats = pats.rename(columns={"Id": "PATIENT"})[["PATIENT", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY", "MARITAL"]]
    pats = pats.merge(splits, on="PATIENT", how="inner")
    pats["BIRTHDATE"] = _d(pats.BIRTHDATE)
    pats["DEATHDATE"] = _d(pats.DEATHDATE)

    enc_cols = ["START", "STOP", "PATIENT", "ENCOUNTERCLASS", "CODE"]
    enc_tv = pd.read_csv(DATA / "train_val/encounters.csv", usecols=enc_cols, dtype=str)
    enc_tv["END"] = _d(enc_tv.STOP).fillna(_d(enc_tv.START))
    end = enc_tv.groupby("PATIENT").END.max().rename("end")
    pats = pats.merge(end, left_on="PATIENT", right_index=True, how="left")
    pats["anchor"] = pats["end"].apply(lambda t: _shift5(t) if pd.notna(t) else pd.NaT)
    ta = pd.read_csv(DATA / "test_anchors.csv", dtype=str).rename(columns={"Id": "PATIENT"})
    ta["anchor_test"] = _d(ta.anchor_date)
    pats = pats.merge(ta[["PATIENT", "anchor_test"]], on="PATIENT", how="left")
    pats["anchor"] = pats["anchor"].fillna(pats["anchor_test"])
    pats = pats.drop(columns=["anchor_test"])
    pats["age"] = (pats.anchor - pats.BIRTHDATE).dt.days / 365.25
    pats["female"] = (pats.GENDER == "F").astype(int)
    pats["died"] = pats.DEATHDATE.notna()
    pats["death_in_window"] = pats.died & (pats.DEATHDATE <= pats["end"])
    pats["anchor_year"] = pats.anchor.dt.year + (pats.anchor.dt.dayofyear - 1) / 365.25

    tables = {
        "conditions": ("START", "COND"), "medications": ("START", "MED"), "procedures": ("DATE", "PROC"),
        "observations": ("DATE", "OBS"), "encounters": ("START", "ENC"), "immunizations": ("DATE", "IMM"),
        "allergies": ("START", "ALLERGY"), "careplans": ("START", "CARE"), "devices": ("START", "DEVICE"),
    }
    frames = []
    for folder in ("train_val", "test"):
        for table, (dcol, kind) in tables.items():
            path = DATA / folder / f"{table}.csv"
            if not path.exists():
                continue
            use = [dcol, "PATIENT", "CODE"]
            if table == "observations":
                use += ["VALUE", "UNITS", "TYPE"]
            if table == "encounters":
                use += ["ENCOUNTERCLASS"]
            df = pd.read_csv(path, usecols=use, dtype=str)
            df = df.rename(columns={dcol: "date"})
            df["kind"] = kind
            df["code"] = df.CODE.astype(str).str.strip()
            if table == "observations":
                df["value"] = df.VALUE.astype("string").str.strip()
                df["units"] = df.UNITS.astype("string").str.strip().fillna("")
                df["num"] = pd.to_numeric(df.VALUE.where(df.TYPE.str.lower() == "numeric"), errors="coerce")
            else:
                df["value"] = pd.NA
                df["units"] = ""
                df["num"] = np.nan
            if table == "encounters":
                df["encclass"] = df.ENCOUNTERCLASS.astype(str)
            else:
                df["encclass"] = pd.NA
            frames.append(df[["PATIENT", "date", "kind", "code", "value", "units", "num", "encclass"]])
    events = pd.concat(frames, ignore_index=True)
    events["date"] = _d(events.date)
    events = events.merge(pats[["PATIENT", "anchor", "split"]], on="PATIENT", how="inner")
    events["years_before"] = (events.anchor - events.date).dt.days / 365.25
    events["pre"] = events.date < events.anchor

    cond = events[events.kind == "COND"]
    first = cond.groupby(["PATIENT", "code"]).date.min().reset_index()
    first_t = first[first.code.isin(codes)].pivot(index="PATIENT", columns="code", values="date")
    first_t = first_t.reindex(index=pats.PATIENT, columns=codes)
    anchor = pats.set_index("PATIENT").anchor
    endv = pats.set_index("PATIENT")["end"]
    A = np.repeat(anchor.values[:, None], len(codes), axis=1)
    E = np.repeat(endv.values[:, None], len(codes), axis=1)
    F = first_t.values
    prior = (F < A) & ~pd.isna(F)
    label = (F >= A) & (F <= E) & ~pd.isna(F)
    Y = pd.DataFrame(label.astype(int), index=pats.PATIENT, columns=codes)
    P = pd.DataFrame(prior.astype(int), index=pats.PATIENT, columns=codes)
    first_t.index.name = "PATIENT"
    out = dict(pats=pats, events=events, first=first, first_targets=first_t, Y=Y, P=P, codes=codes, targets=targets)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


if __name__ == "__main__":
    d = load(force=True)
    print(d["pats"].split.value_counts())
    print(d["events"].shape, d["events"].kind.value_counts().to_dict())
