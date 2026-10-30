"""Join the per-patient genomics and cost features onto the person-period dataset.
Usage: pp_join_extra.py <in_ds> <out_ds> <feature parquet> [<feature parquet> ...]
  e.g. pp_join_extra.py ppds ppds_gcr clean/genomics_features.parquet clean/cost_features_v2.parquet"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
t0 = time.time(); HERE = Path(__file__).resolve().parent
src, dst, feats = HERE / sys.argv[1], HERE / sys.argv[2], [HERE / f for f in sys.argv[3:]]; dst.mkdir(exist_ok=True)
X = pd.read_parquet(src / "X.parquet")
for f in feats:
    G = pd.read_parquet(f); X = X.merge(G, left_on="PATIENT", right_index=True, how="left")
M = dict(np.load(src / "meta.npz", allow_pickle=True)); M["cols"] = np.array([c for c in X.columns if c != "PATIENT"])
X.to_parquet(dst / "X.parquet"); np.savez_compressed(dst / "meta.npz", **M)
print(f"{src.name} + {[f.name for f in feats]} -> {dst.name}: {X.shape} [{time.time()-t0:.0f}s]")
