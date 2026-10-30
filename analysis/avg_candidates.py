"""Average several candidate prediction triplets (oof/val/test) into one new candidate stem (seed averaging of one member).
Usage: avg_candidates.py <out_dir> <out_stem> <dir1:stem1> <dir2:stem2> ...   e.g.
  avg_candidates.py ppds_gcr screen200_d3_all_avg3 ppds_gcr:screen200_d3_all ppds_gcr:screen200_d3_all_s1 ppds_gcr:screen200_d3_all_s2"""
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
out_dir, out_stem, members = sys.argv[1], sys.argv[2], sys.argv[3:]
for split in ("oof", "val", "test"):
    arrs = [np.load(HERE / m.split(":")[0] / f"{m.split(':')[1]}_{split}.npy") for m in members]
    assert all(a.shape == arrs[0].shape for a in arrs), [a.shape for a in arrs]
    np.save(HERE / out_dir / f"{out_stem}_{split}.npy", np.mean(arrs, axis=0))
print(f"averaged {len(members)} members -> {out_dir}/{out_stem}_{{oof,val,test}}.npy")
