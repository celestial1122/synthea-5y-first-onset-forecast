"""Opt-in extra-feature augmentation for the official-row scripts.
EHR_GENOMICS=1 : join clean/genomics_features.parquet, outputs to <dir>_g
EHR_GENOMICS=2 : join genomics + clean/cost_features.parquet (lifetime cost fields decomposed vs pre-anchor cost), outputs to <dir>_gc"""
import os, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
MODE = os.environ.get("EHR_GENOMICS", ""); ON = MODE not in ("", "0"); SFX = {"1": "_g", "2": "_gc", "3": "_gcr", "4": "_gcr3"}.get(MODE, "_g" if ON else "")
def extra():
    G = pd.read_parquet(HERE / "clean" / "genomics_features.parquet")
    if MODE == "2": G = G.join(pd.read_parquet(HERE / "clean" / "cost_features.parquet"), how="left")
    if MODE == "3": G = G.join(pd.read_parquet(HERE / "clean" / "cost_features_v2.parquet"), how="left")
    if MODE == "4": G = G.join(pd.read_parquet(HERE / "clean" / "cost_features_v3.parquet"), how="left")
    return G
def augment(X):
    if not ON: return X
    G = extra()
    if "PATIENT" in X.columns: return X.merge(G, left_on="PATIENT", right_index=True, how="left")
    return X.join(G, how="left")
def outdir(name): return HERE / (name + SFX)
