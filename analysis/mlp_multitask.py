"""Multi-task tabular MLP (shared trunk, 40 heads) on the fused feature matrix (history tokens + numeric summaries +
genomics + cost v3). Two variants: official rows only (mlp_off) and person-period rows (mlp_pp). 5 patient folds x 3
seeds; early stopping on an inner 15% split of the training fold (never on the official val); OOF = honest.
Outputs nn/mlp_{off,pp}_{oof,val,test}.npy (candidate layout)."""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
os.environ["EHR_GENOMICS"] = "4"; import genfeat
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
t0 = time.time(); dev = "cuda"
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"; OUT = HERE / "nn"; OUT.mkdir(exist_ok=True)
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "off"; SEEDS = 3; EPOCHS = 80; PATIENCE = 10
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"}); spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ps = pats.merge(spl, on="PATIENT"); vp = pd.read_csv(HERE / "final/val_predictions_final.csv", dtype={0: str}); tp = pd.read_csv(ROOT / "predictions.csv", dtype={0: str})
ids = {"train": ps.query("split=='train'").PATIENT.values, "val": vp.iloc[:, 0].values, "test": tp.iloc[:, 0].values}
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str)
if VARIANT == "off":
    M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0); codes = list(M["codes"])
    def lab(split, order):
        m = off & (M["split"] == split); pos = pd.Series(np.arange(m.sum()), index=M["pid"][m].astype(str)).reindex(order).values.astype(int)
        return M["y"][m][pos], M["prior"][m][pos].astype(bool)
    X = genfeat.augment(pd.read_parquet(HERE / "clean/X.parquet")); cols = list(X.columns)
    F = {s: X.reindex(ids[s]).values.astype(np.float32) for s in ids}; Y, P = {}, {}
    for s in ids: Y[s], P[s] = lab(s, ids[s])
    pid_tr = ids["train"]; Ftr, Ytr, Ptr = F["train"], Y["train"], P["train"]
else:
    DS = HERE / "ppds_gcr3"; Xp = pd.read_parquet(DS / "X.parquet").set_index("PATIENT"); M = np.load(DS / "meta.npz", allow_pickle=True); codes = list(M["codes"]); cols = list(Xp.columns)
    k = M["k"]; split = M["split"]; pid = M["pid"].astype(str)
    tr = split == "train"; Ftr = Xp.values[tr].astype(np.float32); Ytr = M["y"][tr]; Ptr = M["prior"][tr].astype(bool); pid_tr = pid[tr]
    def offrows(s, order):
        m = (k == 0) & (split == s); pos = pd.Series(np.arange(m.sum()), index=pid[m]).reindex(order).values.astype(int)
        return Xp.values[m][pos].astype(np.float32), M["y"][m][pos], M["prior"][m][pos].astype(bool)
    F, Y, P = {}, {}, {}
    for s in ("val", "test"): F[s], Y[s], P[s] = offrows(s, ids[s])
    F["train"], Y["train"], P["train"] = offrows("train", ids["train"])
grp = o.grp.reindex(codes).values
logcols = np.array([c.startswith(("cnt_", "c5_", "c1_", "nobs_", "cost_")) or c in ("n_events",) for c in cols])
def prep(A, mu=None, sd=None):
    A = A.copy(); A[:, logcols] = np.log1p(np.clip(A[:, logcols], 0, None))
    if mu is None: mu = np.nanmean(A, 0); sd = np.nanstd(A, 0) + 1e-6
    A = (A - mu) / sd; A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0); return np.clip(A, -6, 6).astype(np.float32), mu, sd
class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.f = nn.Sequential(nn.Dropout(0.15), nn.Linear(d, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.35), nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.35), nn.Linear(256, 40))
    def forward(s, x): return s.f(x)
def masked_bce(logit, y, m):
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none"); return (l * m).sum() / m.sum().clamp(min=1)
def fit_predict(Xa, Ya, Ma, Xin, Yin, Min, Xpred, seed):
    torch.manual_seed(seed); np.random.seed(seed); net = Net(Xa.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-2)
    Xa_, Ya_, Ma_ = [torch.tensor(a).to(dev) for a in (Xa, Ya.astype(np.float32), Ma.astype(np.float32))]; Xin_ = torch.tensor(Xin).to(dev)
    best, best_state, bad, ep = -1, None, 0, 0; n = len(Xa_)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]; opt.zero_grad(); loss = masked_bce(net(Xa_[b]), Ya_[b], Ma_[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pin = torch.sigmoid(net(Xin_)).cpu().numpy()
        score = np.mean([ap(Yin[Min[:, j], j], pin[Min[:, j], j]) for j in range(40) if Yin[Min[:, j], j].sum() > 0])
        if score > best: best, bad, best_state = score, 0, {k_: v.clone() for k_, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): return [torch.sigmoid(net(torch.tensor(x).to(dev))).cpu().numpy() for x in Xpred], ep + 1, best
oof = np.zeros((len(ids["train"]), 40)); pv = np.zeros((len(ids["val"]), 40)); pt = np.zeros((len(ids["test"]), 40))
tr_ids = ids["train"]; kf = list(KFold(5, shuffle=True, random_state=0).split(tr_ids))
for fi, (a, b) in enumerate(kf):
    if VARIANT == "off":
        rows_tr = a; row_pids = tr_ids[a]
    else:
        rows_tr = np.where(np.isin(pid_tr, tr_ids[a]))[0]; row_pids = pid_tr[rows_tr]
    rng = np.random.RandomState(fi); inner_pats = rng.choice(tr_ids[a], int(0.15 * len(a)), replace=False); inner = np.isin(row_pids, inner_pats)
    if VARIANT != "off" and "is_official" in cols: inner = inner & (Ftr[rows_tr][:, cols.index("is_official")] > 0.5)
    Xa, mu, sd = prep(Ftr[rows_tr][~inner]); Xin, _, _ = prep(Ftr[rows_tr][inner], mu, sd)
    Xte, _, _ = prep(F["train"][b], mu, sd); Xva, _, _ = prep(F["val"], mu, sd); Xts, _, _ = prep(F["test"], mu, sd)
    Ya, Ma = Ytr[rows_tr][~inner], ~Ptr[rows_tr][~inner]; Yin, Min = Ytr[rows_tr][inner], ~Ptr[rows_tr][inner]
    for seed in range(SEEDS):
        (pte, pva_, pts_), ep, best = fit_predict(Xa, Ya, Ma, Xin, Yin, Min, [Xte, Xva, Xts], seed)
        oof[b] += pte / SEEDS; pv += pva_ / (5 * SEEDS); pt += pts_ / (5 * SEEDS)
    print(f"  fold {fi}: inner-best {best:.4f} at ~{ep} epochs  [{time.time()-t0:.0f}s]", flush=True)
def mAP(Yv, Pv, A): return np.array([ap(Yv[~Pv[:, j], j], A[~Pv[:, j], j]) for j in range(40)])
a_tr = mAP(Y["train"], P["train"], oof); a_va = mAP(Y["val"], P["val"], pv)
print(f"RESULT mlp_{VARIANT}: OOF mAP={a_tr.mean():.4f} | acc {a_tr[grp=='acc'].mean():.3f} / term {a_tr[grp=='term'].mean():.3f} / chain {a_tr[grp=='chain'].mean():.3f} | val {a_va.mean():.4f} [{time.time()-t0:.0f}s]")
for s, arr in (("oof", oof), ("val", pv), ("test", pt)): np.save(OUT / f"mlp_{VARIANT}_{s}.npy", arr)
pd.DataFrame({"cond": o.cond.reindex(codes).str[:28].values, "oof": a_tr.round(3), "val": a_va.round(3)}, index=codes).to_csv(OUT / f"per_condition_mlp_{VARIANT}.csv")
