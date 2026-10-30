"""Build and cache the person-period (discrete-time hazard) dataset once, so that model sweeps do not recompute features.

Rows
  * k = 0  : official anchor rows for train / val / test (labels = official labels for train/val)
  * k >= 1 : landmark rows for TRAIN patients at anchor - k years (step --step), label = first diagnosis in (t, t+5y]
             (fully observed because the training records extend to the record end)
  * --transductive : additionally landmark rows for val/test patients with k >= 5, whose 5-year window lies entirely before
             the anchor (labels observable from the given pre-anchor history alone; no post-anchor information is used)
Features (all computed from events strictly before the landmark): the person_period.py blocks (cnt / c5 / rec / first for
tokens, last / max / min for numerics, static, prior_*) plus the clean-dataset extras (m5 / std / nobs / trend for numerics,
encounter rates by class) and two domain indicators that are constant (=1 / =5) at test time: is_official, yrs_to_end = k + 5.
Output: analysis/ppds/{X.parquet, meta.npz} (meta: y [n,40] int8, prior [n,40] bool, k, pid, split)."""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
from ehr_load import load
ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=2); ap.add_argument("--transductive", action="store_true"); ap.add_argument("--out", default="ppds")
args = ap.parse_args(); t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / args.out; OUT.mkdir(exist_ok=True)
d = load(); pats, events, codes = d["pats"], d["events"], d["codes"]; Ft = d["first_targets"]; pidx = pats.set_index("PATIENT")
ids_by = {s: pats[pats.split == s].PATIENT.values for s in ("train", "val", "test")}
ev = events[["PATIENT", "date", "kind", "code", "value", "units", "num", "encclass", "split"]].copy()
is_obs_cat = (ev.kind == "OBS") & ev.num.isna(); ev["tok"] = ev.kind + ":" + ev.code
ev.loc[is_obs_cat, "tok"] = "OBS:" + ev.loc[is_obs_cat, "code"] + "|" + ev.loc[is_obs_cat, "value"].fillna("").astype(str)
ev.loc[ev.kind == "ENC", "tok"] = "ENC:" + ev.loc[ev.kind == "ENC", "encclass"].astype(str)
ev["is_num"] = (ev.kind == "OBS") & ev.num.notna(); ev.loc[ev.is_num, "code"] = ev.loc[ev.is_num, "code"] + "|" + ev.loc[ev.is_num, "units"].fillna("").astype(str)
pre0 = ev[ev.date < ev.PATIENT.map(pidx.anchor)]
sup = pre0[~pre0.is_num & (pre0.split == "train")][["PATIENT", "tok"]].drop_duplicates().tok.value_counts(); KEEP = sup[sup >= 20].index
nsup = pre0[pre0.is_num & (pre0.split == "train")].groupby("code").PATIENT.nunique(); NKEEP = nsup[nsup >= 50].index
ENCC = sorted(ev.loc[ev.kind == "ENC", "encclass"].dropna().unique())

def features(ids, landmark, k):
    lm = landmark.reindex(ids); e = ev[ev.PATIENT.isin(ids)].copy(); e["lm"] = e.PATIENT.map(lm); e = e[e.date < e.lm]; e["yb"] = (e.lm - e.date).dt.days / 365.25
    sub = e[e.tok.isin(KEEP) & ~e.is_num]; g = sub.groupby(["PATIENT", "tok"])
    fr = [g.size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("cnt_"),
          sub[sub.yb <= 5].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("c5_"),
          sub[sub.yb <= 1].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("c1_"),
          g.yb.min().unstack().reindex(index=ids, columns=KEEP).add_prefix("rec_"), g.yb.max().unstack().reindex(index=ids, columns=KEEP).add_prefix("first_")]
    num = e[e.is_num & e.code.isin(NKEEP)].sort_values("date"); gn = num.groupby(["PATIENT", "code"]).num
    last = gn.last().unstack().reindex(index=ids, columns=NKEEP)
    m5 = num[num.yb <= 5].groupby(["PATIENT", "code"]).num.mean().unstack().reindex(index=ids, columns=NKEEP)
    old = num[num.yb > 2].groupby(["PATIENT", "code"]).num.mean().unstack().reindex(index=ids, columns=NKEEP)
    fr += [last.add_prefix("last_"), gn.max().unstack().reindex(index=ids, columns=NKEEP).add_prefix("max_"), gn.min().unstack().reindex(index=ids, columns=NKEEP).add_prefix("min_"),
           m5.add_prefix("m5_"), gn.std().unstack().reindex(index=ids, columns=NKEEP).add_prefix("std_"), gn.size().unstack().reindex(index=ids, columns=NKEEP).fillna(0).add_prefix("nobs_"),
           (last - old).add_prefix("trend_")]
    # recency of the last numeric observation (years before landmark) -> how fresh the "last_" value is
    fr.append(num.groupby(["PATIENT", "code"]).yb.min().unstack().reindex(index=ids, columns=NKEEP).add_prefix("nrec_"))
    st = pd.DataFrame(index=ids); st["age"] = ((lm - pidx.BIRTHDATE.reindex(ids)).dt.days / 365.25).values; st["female"] = pidx.female.reindex(ids).values
    for r in ("white", "black", "asian", "hispanic"): st[f"race_{r}"] = (pidx.RACE.reindex(ids) == r).astype(int).values
    gp = e.groupby("PATIENT"); st["hist_years"] = gp.yb.max().reindex(ids).fillna(0); st["n_events"] = gp.size().reindex(ids).fillna(0); st["n_events_5y"] = e[e.yb <= 5].groupby("PATIENT").size().reindex(ids).fillna(0)
    st["n_cond_codes"] = e[e.kind == "COND"].groupby("PATIENT").code.nunique().reindex(ids).fillna(0)
    enc = e[e.kind == "ENC"]; er = enc.groupby(["PATIENT", "encclass"]).size().unstack(fill_value=0).reindex(index=ids, columns=ENCC).fillna(0).div(st.hist_years.clip(lower=1.0), axis=0).add_prefix("encrate_")
    er5 = enc[enc.yb <= 5].groupby(["PATIENT", "encclass"]).size().unstack(fill_value=0).reindex(index=ids, columns=ENCC).fillna(0).add_prefix("enc5_")
    st["is_official"] = float(k == 0); st["yrs_to_end"] = float(k + 5)
    Fdt = Ft.reindex(ids); isna = Fdt.isna().values; F = Fdt.values.astype("datetime64[D]").astype(float); F[isna] = np.nan
    L = lm.values.astype("datetime64[D]").astype(float)[:, None]
    prior = (F <= L) & ~np.isnan(F); y = (F > L) & (F <= L + 1826.25) & ~np.isnan(F)
    X = pd.concat([st, pd.DataFrame(prior.astype(np.float32), index=ids, columns=codes).add_prefix("prior_"), er, er5] + fr, axis=1).astype(np.float32)
    return X, y.astype(np.int8), prior

rows = []
all_ids = np.concatenate([ids_by["train"], ids_by["val"], ids_by["test"]])
X0, y0, p0 = features(all_ids, pidx.anchor, 0)
Yoff = d["Y"].reindex(X0.index).values.astype(np.int8); Poff = d["P"].reindex(X0.index).values.astype(bool)
split0 = np.array(["train"] * len(ids_by["train"]) + ["val"] * len(ids_by["val"]) + ["test"] * len(ids_by["test"]))
rows.append((X0, Yoff, Poff, np.zeros(len(all_ids)), split0))
print(f"official rows: {len(X0)} x {X0.shape[1]} [{time.time()-t0:.0f}s]", flush=True)
for k in range(args.step, 60, args.step):
    lm = pidx.anchor - pd.DateOffset(years=k)
    for s in ("train", "val", "test"):
        if s != "train" and (not args.transductive or k < 5): continue
        ids = ids_by[s]; ok = ids[((lm.reindex(ids) - pidx.BIRTHDATE.reindex(ids)).dt.days > 365).values]
        if len(ok) < 30: continue
        Xk, yk, pk = features(ok, lm, k); has = Xk.n_events.values > 0
        rows.append((Xk[has], yk[has], pk[has], np.full(int(has.sum()), k), np.array([s] * int(has.sum()))))
        if s == "train": print(f"  landmark -{k}y ({s}): {int(has.sum())} rows, positives {int(yk[has].sum())} [{time.time()-t0:.0f}s]", flush=True)
    if len(rows[-1][0]) == 0: break
X = pd.concat([r[0] for r in rows]); X.index.name = "PATIENT"
y = np.concatenate([r[1] for r in rows]); prior = np.concatenate([r[2] for r in rows]); k = np.concatenate([r[3] for r in rows]); split = np.concatenate([r[4] for r in rows])
X.reset_index().to_parquet(OUT / "X.parquet"); np.savez_compressed(OUT / "meta.npz", y=y, prior=prior, k=k, split=split, pid=X.index.values.astype(str), codes=np.array(codes), cols=np.array(X.columns))
print(f"saved {len(X)} rows x {X.shape[1]} features -> {OUT}  [{time.time()-t0:.0f}s]  (official train rows {int(((k==0)&(split=='train')).sum())}, landmark rows {int((k>0).sum())})")
