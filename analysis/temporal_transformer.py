"""Temporal Transformer over yearly state vectors (the processed person-period features), i.e. a Transformer whose
tokens are not raw events but the engineered per-year patient state (history counts/recency, numeric summaries, prior
flags, genomics, cost). Causal self-attention over the years; every landmark year carries its own 5-year label
(person-period supervision), the official anchor year gives the prediction. Multi-task 40 heads, masked BCE on at-risk
rows, 5 patient folds x SEEDS seeds, early stopping on an inner 15% split of the training fold (never on val).
Outputs nn/tts_{oof,val,test}.npy (candidate layout). Usage: temporal_transformer.py [maxlen=20] [seeds=3]"""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score as ap
t0 = time.time(); dev = "cuda"
MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 20; SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3; EPOCHS = 60; PATIENCE = 8; D = 96
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"; OUT = HERE / "nn"; OUT.mkdir(exist_ok=True)
DS = HERE / "ppds_gcr"; Xp = pd.read_parquet(DS / "X.parquet"); M = np.load(DS / "meta.npz", allow_pickle=True); codes = list(M["codes"])
pid = M["pid"].astype(str); k = M["k"].astype(int); split = M["split"]; Yall = M["y"]; Pall = M["prior"].astype(bool)
cols = [c for c in Xp.columns if c != "PATIENT"]; F = Xp[cols].values.astype(np.float32); del Xp
logcols = np.array([c.startswith(("cnt_", "c5_", "c1_", "nobs_", "cost_")) for c in cols]); F[:, logcols] = np.log1p(np.clip(F[:, logcols], 0, None))
# ---- build per-patient sequences (oldest landmark first, official anchor last); keep the last MAXLEN years
order = np.lexsort((-k, pid)); pid_o, k_o = pid[order], k[order]
starts = np.r_[0, np.where(pid_o[1:] != pid_o[:-1])[0] + 1]; ends = np.r_[starts[1:], len(pid_o)]
seq = {pid_o[s]: order[s:e][-MAXLEN:] for s, e in zip(starts, ends)}      # row indices, ending with k=0
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"}); spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ps = pats.merge(spl, on="PATIENT"); vp = pd.read_csv(HERE / "final/val_predictions_final.csv", dtype={0: str}); tp = pd.read_csv(ROOT / "predictions.csv", dtype={0: str})
ids = {"train": ps.query("split=='train'").PATIENT.values, "val": vp.iloc[:, 0].values, "test": tp.iloc[:, 0].values}
o = pd.read_csv(HERE / "oracle/per_condition_oracle.csv", index_col=0); o.index = o.index.astype(str); grp = o.grp.reindex(codes).values
def batchify(plist, mu, sd):
    """returns X [B,L,d] standardized, Y [B,L,40], mask [B,L,40] (at-risk & valid), pad [B,L] (True = real step)"""
    # right-padded: real years first (oldest -> official anchor), padding after; so every query position has real keys
    B = len(plist); X = np.zeros((B, MAXLEN, F.shape[1]), np.float32); Yb = np.zeros((B, MAXLEN, 40), np.float32); Mb = np.zeros((B, MAXLEN, 40), np.float32); pad = np.zeros((B, MAXLEN), bool); kk = np.full((B, MAXLEN), 60, np.int64); last = np.zeros(B, np.int64)
    for i, p in enumerate(plist):
        r = seq[p]; L = len(r); X[i, :L] = F[r]; Yb[i, :L] = Yall[r]; Mb[i, :L] = ~Pall[r]; pad[i, :L] = True; kk[i, :L] = np.minimum(k[r], 59); last[i] = L - 1
    X = (X - mu) / sd; X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0); X = np.clip(X, -6, 6); X[~pad] = 0
    return X, Yb, Mb, pad, kk, last
class TTS(nn.Module):
    def __init__(s, d_in):
        super().__init__(); s.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, D), nn.GELU(), nn.LayerNorm(D)); s.pos = nn.Embedding(61, D)
        layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=4 * D, dropout=0.3, batch_first=True, norm_first=True); s.enc = nn.TransformerEncoder(layer, num_layers=2); s.head = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.3), nn.Linear(D, 40))
    def forward(s, x, kk, pad):
        h = s.proj(x) + s.pos(kk); L = x.shape[1]; causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        h = s.enc(h, mask=causal, src_key_padding_mask=~pad); return s.head(h)
def run_fold(train_p, inner_p, pred_sets, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rows_tr = np.concatenate([seq[p] for p in train_p]); mu = np.nanmean(F[rows_tr], 0); sd = np.nanstd(F[rows_tr], 0) + 1e-6
    Xa, Ya, Ma, pa, ka, la = batchify(train_p, mu, sd); Xi, Yi, Mi, pi_, ki, li = batchify(inner_p, mu, sd)
    T = lambda a: torch.tensor(a).to(dev); Xa_, Ya_, Ma_, pa_, ka_ = T(Xa), T(Ya), T(Ma), T(pa), T(ka); Xi_, pi__, ki_, li_ = T(Xi), T(pi_), T(ki), T(li)
    gather_last = lambda out, last_: out[torch.arange(out.shape[0], device=out.device), last_]
    net = TTS(F.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=5e-2); n = len(Xa_); best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            b = perm[i:i + 64]; opt.zero_grad(); logit = net(Xa_[b], ka_[b], pa_[b])
            l = nn.functional.binary_cross_entropy_with_logits(logit, Ya_[b], reduction="none"); m = Ma_[b] * pa_[b].unsqueeze(-1)
            w = torch.where(ka_[b] == 0, 3.0, 1.0).unsqueeze(-1)            # official anchor rows weigh more (they are the test-time rows)
            loss = (l * m * w).sum() / (m * w).sum().clamp(min=1); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad(): pin = torch.sigmoid(gather_last(net(Xi_, ki_, pi__), li_)).cpu().numpy()
        yi, mi = Yi[np.arange(len(li)), li], Mi[np.arange(len(li)), li].astype(bool); score = np.mean([ap(yi[mi[:, j], j], pin[mi[:, j], j]) for j in range(40) if yi[mi[:, j], j].sum() > 0])
        if score > best: best, bad, best_state = score, 0, {k_: v.clone() for k_, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(best_state); net.eval(); outs = []
    with torch.no_grad():
        for plist in pred_sets:
            Xs, _, _, pz, kz, lz = batchify(plist, mu, sd); outs.append(np.concatenate([torch.sigmoid(gather_last(net(T(Xs[i:i + 256]), T(kz[i:i + 256]), T(pz[i:i + 256])), T(lz[i:i + 256]))).cpu().numpy() for i in range(0, len(Xs), 256)]))
    return outs, ep + 1, best
tr_ids = ids["train"]; oof = np.zeros((len(tr_ids), 40)); pv = np.zeros((len(ids["val"]), 40)); pt = np.zeros((len(ids["test"]), 40))
for fi, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr_ids)):
    rng = np.random.RandomState(fi); inner = rng.choice(a, int(0.15 * len(a)), replace=False); outer = np.setdiff1d(a, inner)
    for seed in range(SEEDS):
        (pte, pva_, pts_), ep, best = run_fold(tr_ids[outer], tr_ids[inner], [tr_ids[b], ids["val"], ids["test"]], seed)
        oof[b] += pte / SEEDS; pv += pva_ / (5 * SEEDS); pt += pts_ / (5 * SEEDS)
    print(f"  fold {fi}: inner-best {best:.4f} at ~{ep} epochs [{time.time()-t0:.0f}s]", flush=True)
Ytr = {}; Ptr = {}
for s in ("train", "val"):
    m = (k == 0) & (split == s); pos = pd.Series(np.arange(m.sum()), index=pid[m]).reindex(ids[s]).values.astype(int); Ytr[s], Ptr[s] = Yall[m][pos], Pall[m][pos]
def mAP(Yv, Pv, A): return np.array([ap(Yv[~Pv[:, j], j], A[~Pv[:, j], j]) for j in range(40)])
a_tr, a_va = mAP(Ytr["train"], Ptr["train"], oof), mAP(Ytr["val"], Ptr["val"], pv)
print(f"RESULT tts maxlen={MAXLEN} seeds={SEEDS}: OOF mAP={a_tr.mean():.4f} | acc {a_tr[grp=='acc'].mean():.3f} / term {a_tr[grp=='term'].mean():.3f} / chain {a_tr[grp=='chain'].mean():.3f} | val {a_va.mean():.4f} [{time.time()-t0:.0f}s]")
for s, arr in (("oof", oof), ("val", pv), ("test", pt)): np.save(OUT / f"tts_{s}.npy", arr)
pd.DataFrame({"cond": o.cond.reindex(codes).str[:28].values, "oof": a_tr.round(3), "val": a_va.round(3)}, index=codes).to_csv(OUT / "per_condition_tts.csv")
