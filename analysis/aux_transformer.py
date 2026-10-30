"""Controlled transformer experiments on the OFFICIAL anchors (same windows as the task):
  sparse     : baseline, 40-target BCE at the [PRED] position
  multi      : + auxiliary heads at [PRED]: all first-ever condition codes in the window (>=10 train positives) + death-in-window
  wide       : + wide-and-deep: a linear layer on log1p(bag-of-token counts) added to the 40 logits
  multi_wide : both
Same backbone / budget / early stopping as dense_transformer.py.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from ehr_load import load

PREP = Path(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\artifacts\optimization\prepared_l512")
OUT = Path(__file__).parent / "aux_runs"; OUT.mkdir(exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "multi"
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["17", "42", "73"])]
EPOCHS, PATIENCE, BS, LR, WD = 35, 8, 16, 3e-4, 0.01
PRED_ID, PAD_ID, BOS_ID = 3, 0, 2
AUX_W = 1.0

d = load()
pats = d["pats"].set_index("PATIENT"); codes = d["codes"]; first_all = d["first"]
# auxiliary label space: condition codes with >= 10 first-ever-in-window positives among train patients
tr_ids = d["pats"][d["pats"].split == "train"].PATIENT.values
fa = first_all.merge(d["pats"][["PATIENT", "anchor", "end", "split"]], on="PATIENT")
fa["inwin"] = (fa.date >= fa.anchor) & (fa.date <= fa["end"])
cnt = fa[(fa.split == "train") & fa.inwin].code.value_counts()
AUX = [c for c in cnt[cnt >= 10].index if c not in codes]
print("aux condition codes:", len(AUX), flush=True)
aux_first = fa[fa.code.isin(AUX)].pivot(index="PATIENT", columns="code", values="date").reindex(columns=AUX)


def build(split):
    z = np.load(PREP / f"{split}.npz", allow_pickle=False)
    ids = z["patient_ids"].astype(str); tok = z["tokens"].astype(np.int64); td = z["time_days"].astype(np.float64)
    N, L = tok.shape
    anchor = pd.to_datetime(z["anchor_dates"].astype(str)).values.astype("datetime64[D]").astype(np.int64)
    birth = pats.loc[ids, "BIRTHDATE"].values.astype("datetime64[D]").astype(np.int64)
    female = pats.loc[ids, "female"].values.astype(np.float32)
    lengths = (tok != PAD_ID).sum(1)
    tok2 = np.zeros((N, L + 1), np.int64); day = np.zeros((N, L + 1), np.int64)
    tok2[:, :L] = tok; day[:, :L] = anchor[:, None] - td.astype(np.int64)
    tok2[np.arange(N), lengths] = PRED_ID; day[np.arange(N), lengths] = anchor
    valid2 = tok2 != PAD_ID
    age = (day - birth[:, None]) / 365.25
    prev = np.roll(day, 1, axis=1); prev[:, 0] = day[:, 0]; gap = np.clip(day - prev, 0, None); gap[:, 0] = 0
    feats = np.stack([age / 100.0, np.log1p(gap) / 10.0, np.repeat(female[:, None], L + 1, 1), np.repeat(1 - female[:, None], L + 1, 1)], -1).astype(np.float32)
    feats[~valid2] = 0
    # bag of token counts over the retained sequence (wide path)
    counts = np.zeros((N, V), np.float32)
    for i in range(N):
        c = np.bincount(tok[i][tok[i] > 3], minlength=V); counts[i] = np.log1p(c)
    off_prior = z["prior_conditions"].astype(bool)
    off_y = z["labels"].astype(np.float32) if "labels" in z.files else np.zeros((N, len(codes)), np.float32)
    # auxiliary labels at the official window [anchor, end]
    end = pats.loc[ids, "end"].values.astype("datetime64[D]").astype(np.float64)
    end = np.where(np.isnan(end), anchor + 1826, end)
    af = aux_first.reindex(ids); isna = af.isna().values
    afv = af.values.astype("datetime64[D]").astype(np.float64); afv[isna] = np.inf
    aux_y = ((afv >= anchor[:, None]) & (afv <= end[:, None])).astype(np.float32)
    aux_prior = afv < anchor[:, None]
    died = pats.loc[ids, "died"].values.astype(np.float32)
    aux_y = np.concatenate([aux_y, died[:, None]], 1); aux_mask = np.concatenate([~aux_prior, np.ones((N, 1), bool)], 1)
    return dict(ids=ids, tokens=torch.from_numpy(tok2), feats=torch.from_numpy(feats), counts=torch.from_numpy(counts),
                lengths=torch.from_numpy(lengths), off_y=off_y, off_prior=off_prior,
                y=torch.from_numpy(off_y), mask=torch.from_numpy(~off_prior), aux_y=torch.from_numpy(aux_y), aux_mask=torch.from_numpy(aux_mask))

meta = json.loads((PREP / "metadata.json").read_text()); V = meta["vocab_size"]
data = {s: build(s) for s in ("train", "val", "test")}
LMAX = data["train"]["tokens"].shape[1]; NAUX = data["train"]["aux_y"].shape[1]
tr = data["train"]
print("built; aux targets", NAUX, "| aux positives/patient (train):", float(tr["aux_y"].sum(1).mean()), flush=True)


class Model(nn.Module):
    def __init__(self, vocab, n_targets, n_aux, max_len, d=128, heads=4, layers=3, dropout=0.15, wide=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD_ID); self.pos = nn.Embedding(max_len, d); self.time = nn.Linear(4, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_targets); self.aux_head = nn.Linear(d, n_aux)
        self.wide = nn.Linear(vocab, n_targets) if wide else None
        self.apply(self._init)
        if self.wide is not None:
            nn.init.zeros_(self.wide.weight); nn.init.zeros_(self.wide.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.padding_idx is not None:
                with torch.no_grad(): m.weight[m.padding_idx].zero_()

    def forward(self, tokens, feats, lengths, counts):
        L = tokens.shape[1]
        h = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device))[None] + self.time(feats)
        causal = torch.ones(L, L, device=tokens.device, dtype=torch.bool).triu(1)
        h = self.backbone(self.drop(h), mask=causal, src_key_padding_mask=tokens.eq(PAD_ID), is_causal=True)
        hp = self.drop(self.norm(h[torch.arange(len(lengths), device=tokens.device), lengths]))
        logits = self.head(hp)
        if self.wide is not None:
            logits = logits + self.wide(self.drop(counts))
        return logits, self.aux_head(hp)


def batches(split, bs, shuffle, g):
    n = len(data[split]["ids"]); idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i:i + bs]; L = int(data[split]["lengths"][b].max()) + 1
        out = {}
        for k, v in data[split].items():
            if not torch.is_tensor(v): continue
            out[k] = v[b, :L] if (v.ndim >= 2 and k in ("tokens", "feats")) else v[b]
        yield out


@torch.inference_mode()
def predict(model, split, dev):
    model.eval(); out = []
    for b in batches(split, 64, False, None):
        logits, _ = model(b["tokens"].to(dev), b["feats"].to(dev), b["lengths"].to(dev), b["counts"].to(dev))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    p = np.concatenate(out); p[data[split]["off_prior"]] = 0.0
    return p


def evaluate(p, split):
    y = data[split]["off_y"]
    aps = [average_precision_score(y[:, j], p[:, j]) if y[:, j].sum() else 0.0 for j in range(len(codes))]
    aucs = [roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan for j in range(len(codes))]
    return float(np.mean(aps)), float(np.nanmean(aucs)), aps


def run(seed):
    torch.manual_seed(seed); np.random.seed(seed); dev = torch.device("cuda")
    model = Model(V, len(codes), NAUX, LMAX, wide=MODE in ("wide", "multi_wide")).to(dev)
    pos = tr["off_y"].sum(0); atrisk = (~tr["off_prior"]).sum(0); prior = (pos + 0.5) / (atrisk + 1)
    with torch.no_grad(): model.head.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32, device=dev))
    apos = (tr["aux_y"] * tr["aux_mask"]).sum(0).numpy(); aat = tr["aux_mask"].sum(0).numpy(); aprior = (apos + 0.5) / (aat + 1)
    with torch.no_grad(): model.aux_head.bias.copy_(torch.tensor(np.log(aprior / (1 - aprior)), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    g = torch.Generator().manual_seed(seed); best, best_ep, stale, hist = -1, 0, 0, []; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot, cnt = 0.0, 0
        for b in batches("train", BS, True, g):
            tokens, feats, lengths, counts, y, mask, ay, am = (b[k].to(dev) for k in ("tokens", "feats", "lengths", "counts", "y", "mask", "aux_y", "aux_mask"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, alog = model(tokens, feats, lengths, counts)
            l = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none"); loss = (l * mask).sum() / mask.sum().clamp_min(1)
            if MODE in ("multi", "multi_wide"):
                la = F.binary_cross_entropy_with_logits(alog.float(), ay, reduction="none"); loss = loss + AUX_W * (la * am).sum() / am.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(lengths); cnt += len(lengths)
        sched.step()
        pv = predict(model, "val", dev); m, auc, _ = evaluate(pv, "val"); mtr, _, _ = evaluate(predict(model, "train", dev), "train")
        hist.append(dict(epoch=ep, train_loss=tot / cnt, val_mAP=m, val_auroc=auc, train_mAP=mtr))
        print(f"[{MODE} s{seed}] ep{ep:2d} loss={tot/cnt:.4f} val mAP={m:.4f} auroc={auc:.4f} | train mAP={mtr:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m > best + 1e-6:
            best, best_ep, stale = m, ep, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_val = pv
        else:
            stale += 1
            if stale >= PATIENCE: break
    model.load_state_dict(best_state); pte = predict(model, "test", dev); m, auc, aps = evaluate(best_val, "val")
    np.save(OUT / f"{MODE}_seed{seed}_val.npy", best_val); np.save(OUT / f"{MODE}_seed{seed}_test.npy", pte)
    json.dump(dict(mode=MODE, seed=seed, best_epoch=best_ep, val_mAP=m, val_auroc=auc, per_condition_ap=aps, history=hist), open(OUT / f"{MODE}_seed{seed}.json", "w"), indent=1)
    print(f"### {MODE} seed{seed}: best epoch {best_ep} val mAP={m:.4f} auroc={auc:.4f}", flush=True)
    return m

res = [run(s) for s in SEEDS]
print(f"=== {MODE}: seeds {SEEDS} val mAP mean={np.mean(res):.4f} sd={np.std(res, ddof=1) if len(res)>1 else 0:.4f} ===")
