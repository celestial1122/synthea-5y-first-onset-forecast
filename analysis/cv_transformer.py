"""5-fold cross-validated transformer (multi_wide variant from aux_transformer.py) to obtain honest
out-of-fold predictions on the 2791 training patients. Early stopping uses the official val (model
selection only), never the held-out fold. Outputs: OOF (train), val and test predictions averaged over
fold models, aligned by patient id, so the GBM+transformer blend weight can be chosen on OOF.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from ehr_load import load

PREP = Path(r"C:\Users\NEW PC\Desktop\research_intern_coding_test\artifacts\optimization\prepared_l512")
OUT = Path(__file__).parent / "cv_tf"; OUT.mkdir(exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "multi_wide"
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["17", "42"])]
EPOCHS, PATIENCE, BS, LR, WD, AUX_W = 35, 8, 16, 3e-4, 0.01, 1.0
PRED_ID, PAD_ID = 3, 0

d = load(); pats = d["pats"].set_index("PATIENT"); codes = d["codes"]; first_all = d["first"]
fa = first_all.merge(d["pats"][["PATIENT", "anchor", "end", "split"]], on="PATIENT")
fa["inwin"] = (fa.date >= fa.anchor) & (fa.date <= fa["end"])
cnt = fa[(fa.split == "train") & fa.inwin].code.value_counts(); AUX = [c for c in cnt[cnt >= 10].index if c not in codes]
aux_first = fa[fa.code.isin(AUX)].pivot(index="PATIENT", columns="code", values="date").reindex(columns=AUX)
meta = json.loads((PREP / "metadata.json").read_text()); V = meta["vocab_size"]

def build(split):
    z = np.load(PREP / f"{split}.npz", allow_pickle=False)
    ids = z["patient_ids"].astype(str); tok = z["tokens"].astype(np.int64); td = z["time_days"].astype(np.float64); N, L = tok.shape
    anchor = pd.to_datetime(z["anchor_dates"].astype(str)).values.astype("datetime64[D]").astype(np.int64)
    birth = pats.loc[ids, "BIRTHDATE"].values.astype("datetime64[D]").astype(np.int64); female = pats.loc[ids, "female"].values.astype(np.float32)
    lengths = (tok != PAD_ID).sum(1)
    tok2 = np.zeros((N, L + 1), np.int64); day = np.zeros((N, L + 1), np.int64); tok2[:, :L] = tok; day[:, :L] = anchor[:, None] - td.astype(np.int64)
    tok2[np.arange(N), lengths] = PRED_ID; day[np.arange(N), lengths] = anchor; valid2 = tok2 != PAD_ID
    age = (day - birth[:, None]) / 365.25; prev = np.roll(day, 1, axis=1); prev[:, 0] = day[:, 0]; gap = np.clip(day - prev, 0, None); gap[:, 0] = 0
    feats = np.stack([age / 100.0, np.log1p(gap) / 10.0, np.repeat(female[:, None], L + 1, 1), np.repeat(1 - female[:, None], L + 1, 1)], -1).astype(np.float32); feats[~valid2] = 0
    counts = np.zeros((N, V), np.float32)
    for i in range(N): counts[i] = np.log1p(np.bincount(tok[i][tok[i] > 3], minlength=V))
    off_prior = z["prior_conditions"].astype(bool); off_y = z["labels"].astype(np.float32) if "labels" in z.files else np.zeros((N, len(codes)), np.float32)
    end = pats.loc[ids, "end"].values.astype("datetime64[D]").astype(np.float64); end = np.where(np.isnan(end), anchor + 1826, end)
    af = aux_first.reindex(ids); isna = af.isna().values; afv = af.values.astype("datetime64[D]").astype(np.float64); afv[isna] = np.inf
    aux_y = np.concatenate([((afv >= anchor[:, None]) & (afv <= end[:, None])).astype(np.float32), pats.loc[ids, "died"].values.astype(np.float32)[:, None]], 1)
    aux_mask = np.concatenate([~(afv < anchor[:, None]), np.ones((N, 1), bool)], 1)
    return dict(ids=ids, tokens=torch.from_numpy(tok2), feats=torch.from_numpy(feats), counts=torch.from_numpy(counts), lengths=torch.from_numpy(lengths),
                off_y=off_y, off_prior=off_prior, y=torch.from_numpy(off_y), mask=torch.from_numpy(~off_prior), aux_y=torch.from_numpy(aux_y), aux_mask=torch.from_numpy(aux_mask))

full = {s: build(s) for s in ("train", "val", "test")}
LMAX = full["train"]["tokens"].shape[1]; NAUX = full["train"]["aux_y"].shape[1]
def subset(ds, idx):
    return {k: (v[idx] if torch.is_tensor(v) else (v[idx] if isinstance(v, np.ndarray) else v)) for k, v in ds.items()}


class Model(nn.Module):
    def __init__(self, vocab, n_targets, n_aux, max_len, d=128, heads=4, layers=3, dropout=0.15, wide=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD_ID); self.pos = nn.Embedding(max_len, d); self.time = nn.Linear(4, d); self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_targets); self.aux_head = nn.Linear(d, n_aux); self.wide = nn.Linear(vocab, n_targets) if wide else None
        self.apply(self._init)
        if self.wide is not None: nn.init.zeros_(self.wide.weight); nn.init.zeros_(self.wide.bias)
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
        L = tokens.shape[1]; h = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device))[None] + self.time(feats)
        causal = torch.ones(L, L, device=tokens.device, dtype=torch.bool).triu(1)
        h = self.backbone(self.drop(h), mask=causal, src_key_padding_mask=tokens.eq(PAD_ID), is_causal=True)
        hp = self.drop(self.norm(h[torch.arange(len(lengths), device=tokens.device), lengths])); logits = self.head(hp)
        if self.wide is not None: logits = logits + self.wide(self.drop(counts))
        return logits, self.aux_head(hp)


def batches(ds, bs, shuffle, g):
    n = len(ds["ids"]); idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i:i + bs]; L = int(ds["lengths"][b].max()) + 1
        yield {k: (v[b, :L] if (v.ndim >= 2 and k in ("tokens", "feats")) else v[b]) for k, v in ds.items() if torch.is_tensor(v)}

@torch.inference_mode()
def predict(model, ds, dev):
    model.eval(); out = []
    for b in batches(ds, 64, False, None):
        logits, _ = model(b["tokens"].to(dev), b["feats"].to(dev), b["lengths"].to(dev), b["counts"].to(dev)); out.append(torch.sigmoid(logits.float()).cpu().numpy())
    p = np.concatenate(out); p[ds["off_prior"]] = 0.0; return p

def mAP(p, ds):
    y = ds["off_y"]; return float(np.mean([average_precision_score(y[:, j], p[:, j]) if y[:, j].sum() else 0.0 for j in range(len(codes))]))

def train_one(trn, seed):
    torch.manual_seed(seed); np.random.seed(seed); dev = torch.device("cuda")
    model = Model(V, len(codes), NAUX, LMAX, wide=MODE in ("wide", "multi_wide")).to(dev)
    pos = trn["off_y"].sum(0); atrisk = (~trn["off_prior"]).sum(0); prior = (pos + 0.5) / (atrisk + 1)
    with torch.no_grad(): model.head.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32, device=dev))
    apos = (trn["aux_y"] * trn["aux_mask"]).sum(0).numpy(); aat = trn["aux_mask"].sum(0).numpy(); aprior = (apos + 0.5) / (aat + 1)
    with torch.no_grad(): model.aux_head.bias.copy_(torch.tensor(np.log(aprior / (1 - aprior)), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    g = torch.Generator().manual_seed(seed); best, stale = -1, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in batches(trn, BS, True, g):
            tokens, feats, lengths, counts, y, mask, ay, am = (b[k].to(dev) for k in ("tokens", "feats", "lengths", "counts", "y", "mask", "aux_y", "aux_mask"))
            with torch.autocast("cuda", dtype=torch.bfloat16): logits, alog = model(tokens, feats, lengths, counts)
            l = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none"); loss = (l * mask).sum() / mask.sum().clamp_min(1)
            if MODE in ("multi", "multi_wide"):
                la = F.binary_cross_entropy_with_logits(alog.float(), ay, reduction="none"); loss = loss + AUX_W * (la * am).sum() / am.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        m = mAP(predict(model, full["val"], dev), full["val"])
        if m > best + 1e-6: best, stale, best_state = m, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE: break
    model.load_state_dict(best_state); return model, best

dev = torch.device("cuda"); t0 = time.time()
ntr = len(full["train"]["ids"]); oof = np.zeros((ntr, len(codes))); pval = np.zeros((len(full["val"]["ids"]), len(codes))); pte = np.zeros((len(full["test"]["ids"]), len(codes))); nmodels = 0
for seed in SEEDS:
    for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(ntr))):
        model, best = train_one(subset(full["train"], torch.from_numpy(a)), seed * 100 + fold)
        held = subset(full["train"], torch.from_numpy(b)); po = predict(model, held, dev)
        oof[b] += po / len(SEEDS); pval += predict(model, full["val"], dev); pte += predict(model, full["test"], dev); nmodels += 1
        print(f"[{MODE}] seed{seed} fold{fold}: val-selected mAP={best:.4f} | held-out fold mAP={mAP(po, held):.4f}  ({time.time()-t0:.0f}s)", flush=True)
pval /= nmodels; pte /= nmodels
print(f"=== {MODE} {len(SEEDS)} seeds x 5 folds: OOF mAP (2791) = {mAP(oof, full['train']):.4f} | fold-ensemble val mAP = {mAP(pval, full['val']):.4f} ===")
for name, p, ds in (("oof", oof, full["train"]), ("val", pval, full["val"]), ("test", pte, full["test"])):
    pd.DataFrame(p, index=pd.Index(ds["ids"], name="patient_id"), columns=codes).to_csv(OUT / f"{MODE}_{name}.csv")
print("saved", OUT)
