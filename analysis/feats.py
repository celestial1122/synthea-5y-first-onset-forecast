"""Shared tabular feature builder (official anchors), identical to tabular_baseline.py / cv_all.py."""
import numpy as np, pandas as pd
from ehr_load import load


def build_features():
    d = load(); pats, events, codes = d["pats"], d["events"], d["codes"]
    pidx = pats.set_index("PATIENT"); ids = pats.PATIENT.values
    ev = events[["PATIENT", "date", "kind", "code", "value", "num", "encclass", "split"]].copy()
    is_obs_cat = (ev.kind == "OBS") & ev.num.isna()
    ev["tok"] = ev.kind + ":" + ev.code
    ev.loc[is_obs_cat, "tok"] = "OBS:" + ev.loc[is_obs_cat, "code"] + "|" + ev.loc[is_obs_cat, "value"].fillna("").astype(str)
    ev.loc[ev.kind == "ENC", "tok"] = "ENC:" + ev.loc[ev.kind == "ENC", "encclass"].astype(str)
    ev["is_num"] = (ev.kind == "OBS") & ev.num.notna()
    pre = ev[ev.date < ev.PATIENT.map(pidx.anchor)].copy()
    pre["yb"] = (pre.PATIENT.map(pidx.anchor) - pre.date).dt.days / 365.25
    sup = pre[~pre.is_num & (pre.split == "train")][["PATIENT", "tok"]].drop_duplicates().tok.value_counts(); KEEP = sup[sup >= 20].index
    nsup = pre[pre.is_num & (pre.split == "train")].groupby("code").PATIENT.nunique(); NKEEP = nsup[nsup >= 50].index
    sub = pre[pre.tok.isin(KEEP) & ~pre.is_num]; g = sub.groupby(["PATIENT", "tok"])
    frames = [g.size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("cnt_"),
              sub[sub.yb <= 5].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("c5_"),
              sub[sub.yb <= 1].groupby(["PATIENT", "tok"]).size().unstack(fill_value=0).reindex(index=ids, columns=KEEP).fillna(0).add_prefix("c1_"),
              g.yb.min().unstack().reindex(index=ids, columns=KEEP).add_prefix("rec_"),
              g.yb.max().unstack().reindex(index=ids, columns=KEEP).add_prefix("first_")]
    num = pre[pre.is_num & pre.code.isin(NKEEP)].sort_values("date"); gn = num.groupby(["PATIENT", "code"]).num
    frames += [gn.last().unstack().reindex(index=ids, columns=NKEEP).add_prefix("last_"), gn.max().unstack().reindex(index=ids, columns=NKEEP).add_prefix("max_"),
               gn.min().unstack().reindex(index=ids, columns=NKEEP).add_prefix("min_"),
               num[num.yb <= 5].groupby(["PATIENT", "code"]).num.mean().unstack().reindex(index=ids, columns=NKEEP).add_prefix("m5_")]
    st = pd.DataFrame(index=ids); st["age"] = pidx.age.reindex(ids); st["female"] = pidx.female.reindex(ids)
    gp = pre.groupby("PATIENT"); st["hist_years"] = gp.yb.max().reindex(ids).fillna(0); st["n_events"] = gp.size().reindex(ids).fillna(0)
    st["n_events_5y"] = pre[pre.yb <= 5].groupby("PATIENT").size().reindex(ids).fillna(0)
    st["n_cond_codes"] = pre[pre.kind == "COND"].groupby("PATIENT").code.nunique().reindex(ids).fillna(0)
    st["n_med_codes"] = pre[pre.kind == "MED"].groupby("PATIENT").code.nunique().reindex(ids).fillna(0)
    st["n_tokens"] = pre.groupby("PATIENT").tok.nunique().reindex(ids).fillna(0)
    for r in ("white", "black", "asian", "hispanic"): st[f"race_{r}"] = (pidx.RACE.reindex(ids) == r).astype(int)
    X = pd.concat([st, d["P"].add_prefix("prior_")] + frames, axis=1).astype(np.float32)
    return d, X
