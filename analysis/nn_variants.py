"""Two targeted tests of 'is it the input format / can NN and XGB be combined' (official rows, fused features, 5 patient
folds x 3 seeds, inner-split early stopping, honest OOF):
  ple   : piecewise-linear quantile encoding of every feature (Gorishniy et al. 2022) with a learnable per-feature
          piecewise-linear transform in front of the MLP, so thresholds are as easy for the net as for a tree.
  resid : NN trained on top of the best XGB candidate's logit as a fixed offset (boosting-style residual learner), so the
          net only has to model what the trees missed. Offset = ppds_gcr:screen200_d3_all (OOF for train rows).
Outputs nn/{ple,resid}_{oof,val,test}.npy."""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
os.environ["EHR_GENOMICS"] = "4"; import genfeat
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
t0 = time.time(); dev = "cuda"; MODE = sys.argv[1] if len(sys.argv) > 1 else "ple"; SEEDS = 3; EPOCHS = 80; PATIENCE = 10; B = 8
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"; OUT = HERE / "nn"; OUT.mkdir(exist_ok=True)
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"}); spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ps = pats.merge(spl, on="PATIENT"); vp = pd.read_csv(HERE / "final/val_predictions_final.csv", dtype={0: str}); tp = pd.read_csv(ROOT / "predictions.csv", dtype={0: str})
ids = {"train": ps.query("split=='train'").PATIENT.values, "val": vp.iloc[:, 0].values, "test": tp.iloc[:, 0].values}
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0); codes = list(M["codes"])
def lab(split, order):
    m = off & (M["split"] == split); pos = pd.Series(np.arange(m.sum()), index=M["pid"][m].astype(str)).reindex(order).values.astype(int)
    return M["y"][m][pos], M["prior"][m][pos].astype(bool)
X = genfeat.augment(pd.read_parquet(HERE / "clean/X.parquet")); cols = list(X.columns)
F = {s: X.reindex(ids[s]).values.astype(np.float32) for s in ids}; Y, P = {}, {}
for s in ids: Y[s], P[s] = lab(s, ids[s])
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str); grp = o.grp.reindex(codes).values
logcols = np.array([c.startswith(("cnt_", "c5_", "c1_", "nobs_", "cost_")) for c in cols])
def prep(A, mu=None, sd=None):
    A = A.copy(); A[:, logcols] = np.log1p(np.clip(A[:, logcols], 0, None))
    if mu is None: mu = np.nanmean(A, 0); sd = np.nanstd(A, 0) + 1e-6
    A = (A - mu) / sd; A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0); return np.clip(A, -6, 6).astype(np.float32), mu, sd
# ---- offsets for the residual mode
OFF = None
if MODE == "resid":
    OFF = {"train": np.load(HERE / "ppds_gcr/screen200_d3_all_oof.npy"), "val": np.load(HERE / "ppds_gcr/screen200_d3_all_val.npy"), "test": np.load(HERE / "ppds_gcr/screen200_d3_all_test.npy")}
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).astype(np.float32); OFF = {s: lg(v) for s, v in OFF.items()}
class PLE(nn.Module):
    """per-feature piecewise-linear encoding on quantile edges followed by a learnable per-feature linear map back to 1 dim"""
    def __init__(s, edges):   # edges: d x (B+1)
        super().__init__(); s.register_buffer("lo", torch.tensor(edges[:, :-1])); s.register_buffer("hi", torch.tensor(edges[:, 1:])); d = edges.shape[0]
        s.w = nn.Parameter(torch.randn(d, B) * 0.1); s.b = nn.Parameter(torch.zeros(d))
    def forward(s, x):   # x: n x d
        width = (s.hi - s.lo).clamp(min=1e-6); t = ((x.unsqueeze(-1) - s.lo) / width).clamp(0, 1)   # n x d x B
        return (t * s.w).sum(-1) + s.b
class Net(nn.Module):
    def __init__(s, d, edges=None, small=False):
        super().__init__(); s.ple = PLE(edges) if edges is not None else None
        h1, h2 = (256, 64) if small else (512, 256)
        s.f = nn.Sequential(nn.Dropout(0.15), nn.Linear(d, h1), nn.GELU(), nn.LayerNorm(h1), nn.Dropout(0.35), nn.Linear(h1, h2), nn.GELU(), nn.LayerNorm(h2), nn.Dropout(0.35), nn.Linear(h2, 40))
        if small: nn.init.zeros_(s.f[-1].weight); nn.init.zeros_(s.f[-1].bias)
    def forward(s, x): return s.f(s.ple(x) if s.ple is not None else x)
def masked_bce(logit, y, m):
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none"); return (l * m).sum() / m.sum().clamp(min=1)
def fit_predict(Xa, Ya, Ma, Oa, Xin, Yin, Min, Oin, Xpred, Opred, seed, edges):
    torch.manual_seed(seed); np.random.seed(seed); small = MODE == "resid"
    net = Net(Xa.shape[1], edges, small).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=5e-4 if small else 1e-3, weight_decay=5e-2 if small else 2e-2)
    T = lambda a: torch.tensor(a).to(dev); Xa_, Ya_, Ma_, Xin_ = T(Xa), T(Ya.astype(np.float32)), T(Ma.astype(np.float32)), T(Xin)
    Oa_ = T(Oa) if Oa is not None else 0; Oin_ = T(Oin) if Oin is not None else 0
    best, best_state, bad, ep = -1, None, 0, 0; n = len(Xa_)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]; opt.zero_grad(); loss = masked_bce(net(Xa_[b]) + (Oa_[b] if Oa is not None else 0), Ya_[b], Ma_[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pin = torch.sigmoid(net(Xin_) + Oin_).cpu().numpy()
        score = np.mean([ap(Yin[Min[:, j], j], pin[Min[:, j], j]) for j in range(40) if Yin[Min[:, j], j].sum() > 0])
        if score > best: best, bad, best_state = score, 0, {k_: v.clone() for k_, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): return [torch.sigmoid(net(T(x)) + (T(o_) if o_ is not None else 0)).cpu().numpy() for x, o_ in zip(Xpred, Opred)], ep + 1, best
def make_edges(A):
    qs = np.linspace(0, 1, B + 1); E = np.nanquantile(A, qs, axis=0).T.astype(np.float32); E = np.nan_to_num(E, nan=0.0)
    return np.maximum.accumulate(E, axis=1)
oof = np.zeros((len(ids["train"]), 40)); pv = np.zeros((len(ids["val"]), 40)); pt = np.zeros((len(ids["test"]), 40))
tr_ids = ids["train"]
for fi, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr_ids)):
    rng = np.random.RandomState(fi); inner_pats = rng.choice(tr_ids[a], int(0.15 * len(a)), replace=False); inner = np.isin(tr_ids[a], inner_pats)
    ra, rin = a[~inner], a[inner]
    Xa, mu, sd = prep(F["train"][ra]); Xin, _, _ = prep(F["train"][rin], mu, sd); Xte, _, _ = prep(F["train"][b], mu, sd); Xva, _, _ = prep(F["val"], mu, sd); Xts, _, _ = prep(F["test"], mu, sd)
    edges = make_edges(Xa) if MODE == "ple" else None
    Oa, Oin, Ote, Ova, Ots = (OFF["train"][ra], OFF["train"][rin], OFF["train"][b], OFF["val"], OFF["test"]) if OFF is not None else (None,) * 5
    for seed in range(SEEDS):
        (pte, pva_, pts_), ep, best = fit_predict(Xa, Y["train"][ra], ~P["train"][ra], Oa, Xin, Y["train"][rin], ~P["train"][rin], Oin, [Xte, Xva, Xts], [Ote, Ova, Ots], seed, edges)
        oof[b] += pte / SEEDS; pv += pva_ / (5 * SEEDS); pt += pts_ / (5 * SEEDS)
    print(f"  fold {fi}: inner-best {best:.4f} at ~{ep} epochs  [{time.time()-t0:.0f}s]", flush=True)
def mAP(Yv, Pv, A): return np.array([ap(Yv[~Pv[:, j], j], A[~Pv[:, j], j]) for j in range(40)])
a_tr = mAP(Y["train"], P["train"], oof); a_va = mAP(Y["val"], P["val"], pv)
line = f"RESULT nn_{MODE}: OOF mAP={a_tr.mean():.4f} | acc {a_tr[grp=='acc'].mean():.3f} / term {a_tr[grp=='term'].mean():.3f} / chain {a_tr[grp=='chain'].mean():.3f} | val {a_va.mean():.4f} [{time.time()-t0:.0f}s]"
if OFF is not None:
    sg = lambda z: 1 / (1 + np.exp(-z)); b_tr = mAP(Y["train"], P["train"], sg(OFF["train"])); b_va = mAP(Y["val"], P["val"], sg(OFF["val"]))
    line += f"\n   offset model alone: OOF {b_tr.mean():.4f} | val {b_va.mean():.4f}  -> residual NN delta OOF {a_tr.mean()-b_tr.mean():+.4f}, val {a_va.mean()-b_va.mean():+.4f}"
print(line)
for s, arr in (("oof", oof), ("val", pv), ("test", pt)): np.save(OUT / f"{MODE}_{s}.npy", arr)
pd.DataFrame({"cond": o.cond.reindex(codes).str[:28].values, "oof": a_tr.round(3), "val": a_va.round(3)}, index=codes).to_csv(OUT / f"per_condition_nn_{MODE}.csv")
