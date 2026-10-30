"""Controlled experiment: one label per patient (sparse, = current paradigm) vs dense multi-horizon labels at
every timestep (every event position predicts the next-5-year first diagnoses), same backbone, same budget.

Leak control: time features are age-at-event, gap since previous event, sex. No anchor-relative times and no
static age-at-anchor, so hidden state at position t only depends on events <= t. A [PRED] token (id 3) is
appended at the official anchor date; official val/test predictions are read there.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from ehr_load import load

PREP = Path(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\artifacts\optimization\prepared_l512")
OUT = Path(__file__).parent / "dense_runs"; OUT.mkdir(exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "dense"      # sparse | dense | dense_mix
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["17", "42", "73"])]
EPOCHS, PATIENCE, BS, LR, WD = 35, 8, 16, 3e-4, 0.01
PRED_ID, PAD_ID, BOS_ID = 3, 0, 2
WINDOW = 1826  # days

d = load()
pats = d["pats"].set_index("PATIENT")
Ft = d["first_targets"]            # per patient first date of each target
codes = d["codes"]

def build(split):
    z = np.load(PREP / f"{split}.npz", allow_pickle=False)
    ids = z["patient_ids"].astype(str); tok = z["tokens"].astype(np.int64); td = z["time_days"].astype(np.float64)
    N, L = tok.shape
    anchor = pd.to_datetime(z["anchor_dates"].astype(str)).values.astype("datetime64[D]").astype(np.int64)
    birth = pats.loc[ids, "BIRTHDATE"].values.astype("datetime64[D]").astype(np.int64)
    female = pats.loc[ids, "female"].values.astype(np.float32)
    Fdt = Ft.loc[ids]
    isna = Fdt.isna().values                                  # NaT must become +inf (never diagnosed)
    first = Fdt.values.astype("datetime64[D]").astype(np.int64).astype(np.float64)
    first[isna] = np.inf
    assert np.isinf(first).mean() > 0.5, "most patient/target cells should be never-diagnosed"
    valid = tok != PAD_ID
    lengths = valid.sum(1)
    # append PRED token at the anchor
    tok2 = np.zeros((N, L + 1), np.int64); day = np.zeros((N, L + 1), np.int64)
    tok2[:, :L] = tok
    day[:, :L] = anchor[:, None] - td.astype(np.int64)
    tok2[np.arange(N), lengths] = PRED_ID
    day[np.arange(N), lengths] = anchor
    valid2 = tok2 != PAD_ID
    is_pred = tok2 == PRED_ID
    # time features: age at event, gap since previous event, sex
    age = (day - birth[:, None]) / 365.25
    prev = np.roll(day, 1, axis=1); prev[:, 0] = day[:, 0]
    gap = np.clip(day - prev, 0, None)
    gap[:, 0] = 0
    feats = np.stack([age / 100.0, np.log1p(gap) / 10.0, np.repeat(female[:, None], L + 1, 1), np.repeat(1 - female[:, None], L + 1, 1)], -1).astype(np.float32)
    feats[~valid2] = 0
    # dense labels / prior for every position: window (day, day+WINDOW]
    dense_y = (first[:, None, :] > day[:, :, None]) & (first[:, None, :] <= day[:, :, None] + WINDOW)
    dense_prior = first[:, None, :] <= day[:, :, None]
    dense_mask = valid2[:, :, None] & ~dense_prior
    dense_mask[:, 0, :] = False                           # BOS carries no history
    # official labels at PRED
    off_prior = z["prior_conditions"].astype(bool)
    off_y = z["labels"].astype(np.float32) if "labels" in z.files else np.zeros((N, len(codes)), np.float32)
    rows = np.arange(N)
    dense_y = dense_y.astype(np.float32)
    dense_y[rows, lengths] = off_y
    dense_mask[rows, lengths] = valid2[rows, lengths][:, None] & ~off_prior
    return dict(ids=ids, tokens=torch.from_numpy(tok2), feats=torch.from_numpy(feats), y=torch.from_numpy(dense_y),
                mask=torch.from_numpy(dense_mask), is_pred=torch.from_numpy(is_pred), lengths=torch.from_numpy(lengths),
                off_y=off_y, off_prior=off_prior)

data = {s: build(s) for s in ("train", "val", "test")}
meta = json.loads((PREP / "metadata.json").read_text())
V = meta["vocab_size"]; LMAX = data["train"]["tokens"].shape[1]
print("built", {s: tuple(v["tokens"].shape) for s, v in data.items()}, "vocab", V, flush=True)
# sanity: dense labels at PRED equal official labels; positive density
tr = data["train"]
print("train dense supervised cells:", int(tr["mask"].sum()), "positives:", int((tr["y"] * tr["mask"]).sum()),
      "| PRED cells:", int(tr["mask"][torch.arange(len(tr["ids"])), tr["lengths"]].sum()))


class Model(nn.Module):
    def __init__(self, vocab, n_targets, max_len, d=128, heads=4, layers=3, dropout=0.15):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d)
        self.time = nn.Linear(4, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_targets)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.padding_idx is not None:
                with torch.no_grad(): m.weight[m.padding_idx].zero_()

    def forward(self, tokens, feats):
        L = tokens.shape[1]
        h = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device))[None] + self.time(feats)
        causal = torch.ones(L, L, device=tokens.device, dtype=torch.bool).triu(1)
        h = self.backbone(self.drop(h), mask=causal, src_key_padding_mask=tokens.eq(PAD_ID), is_causal=True)
        return self.head(self.drop(self.norm(h)))      # (B, L, 40) logits at every position


def batches(split, bs, shuffle, g):
    n = len(data[split]["ids"]); idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i:i + bs]; L = int(data[split]["lengths"][b].max()) + 1
        yield {k: v[b, :L] if torch.is_tensor(v) and v.ndim >= 2 else (v[b] if torch.is_tensor(v) else None) for k, v in data[split].items()}


@torch.inference_mode()
def predict(model, split, dev):
    model.eval(); out = []
    for b in batches(split, 64, False, None):
        logits = model(b["tokens"].to(dev), b["feats"].to(dev)).float()
        rows = torch.arange(len(b["lengths"]), device=dev)
        p = torch.sigmoid(logits[rows, b["lengths"].to(dev)])
        out.append(p.cpu().numpy())
    p = np.concatenate(out); p[data[split]["off_prior"]] = 0.0
    return p


def evaluate(p, split):
    y = data[split]["off_y"]; aps, aucs = [], []
    for j in range(len(codes)):
        aps.append(average_precision_score(y[:, j], p[:, j]) if y[:, j].sum() else 0.0)
        aucs.append(roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan)
    return float(np.mean(aps)), float(np.nanmean(aucs)), aps


def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda")
    model = Model(V, len(codes), LMAX).to(dev)
    pos = tr["off_y"].sum(0); atrisk = (~tr["off_prior"]).sum(0); prior = (pos + 0.5) / (atrisk + 1)
    with torch.no_grad(): model.head.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, stale, hist = -1, 0, 0, []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot, cnt = 0.0, 0
        for b in batches("train", BS, True, g):
            tokens, feats, y, mask = (b[k].to(dev) for k in ("tokens", "feats", "y", "mask"))
            rows = torch.arange(len(b["lengths"]), device=dev); lens = b["lengths"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens, feats)
            logits = logits.float()
            losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            pred_mask = torch.zeros_like(mask); pred_mask[rows, lens] = mask[rows, lens]
            l_pred = (losses * pred_mask).sum() / pred_mask.sum().clamp_min(1)
            if MODE == "sparse":
                loss = l_pred
            elif MODE == "dense":
                loss = (losses * mask).sum() / mask.sum().clamp_min(1)
            else:  # dense_mix: half weight on the official anchor position
                loss = 0.5 * (losses * mask).sum() / mask.sum().clamp_min(1) + 0.5 * l_pred
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(b["lengths"]); cnt += len(b["lengths"])
        sched.step()
        pv = predict(model, "val", dev); m, auc, _ = evaluate(pv, "val")
        ptr = predict(model, "train", dev); mtr, _, _ = evaluate(ptr, "train")
        hist.append(dict(epoch=ep, train_loss=tot / cnt, val_mAP=m, val_auroc=auc, train_mAP_at_pred=mtr, sec=time.time() - t0))
        print(f"[{MODE} s{seed}] ep{ep:2d} loss={tot/cnt:.4f} val mAP={m:.4f} auroc={auc:.4f} | train mAP@PRED={mtr:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m > best + 1e-6:
            best, best_ep, stale = m, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val = pv
        else:
            stale += 1
            if stale >= PATIENCE: break
    model.load_state_dict(best_state)
    pte = predict(model, "test", dev)
    m, auc, aps = evaluate(best_val, "val")
    np.save(OUT / f"{MODE}_seed{seed}_val.npy", best_val); np.save(OUT / f"{MODE}_seed{seed}_test.npy", pte)
    json.dump(dict(mode=MODE, seed=seed, best_epoch=best_ep, val_mAP=m, val_auroc=auc, per_condition_ap=aps, history=hist),
              open(OUT / f"{MODE}_seed{seed}.json", "w"), indent=1)
    print(f"### {MODE} seed{seed}: best epoch {best_ep} val mAP={m:.4f} auroc={auc:.4f}", flush=True)
    return m


res = [run(s) for s in SEEDS]
print(f"=== {MODE}: seeds {SEEDS} val mAP mean={np.mean(res):.4f} sd={np.std(res, ddof=1) if len(res)>1 else 0:.4f} ===")
