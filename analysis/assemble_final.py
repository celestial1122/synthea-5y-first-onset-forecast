"""Final assembly by Caruana-style greedy ensemble selection per target on OOF predictions (Caruana et al. 2004),
cross-fitted at the patient level (select on half A of the OOF rows, evaluate on half B, and vice versa) so the reported
OOF number is honest. Candidates = every model family that produced OOF/val/test predictions for the official anchors.
Writes predictions_final.csv, analysis/final/weights.csv, analysis/final/summary.json."""
import json, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from ehr_load import load
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; OUT = HERE / "final"; OUT.mkdir(exist_ok=True)
d = load(); pats, Y, P, codes, targets = d["pats"], d["Y"], d["P"], d["codes"], d["targets"]; desc = dict(zip(targets.CODE, targets.DESCRIPTION))
ids = {s: pats[pats.split == s].PATIENT.values for s in ("train", "val", "test")}
Yv = {s: Y.loc[ids[s]].values for s in ("train", "val")}; Pv = {s: P.loc[ids[s]].values.astype(bool) for s in ("train", "val", "test")}
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
GROUPS = [["162573006", "424132000", "254637007"], ["22325002", "723857007"]]

def npy(prefix_dir, stem):
    f = {s: HERE / prefix_dir / f"{stem}{k}.npy" for s, k in (("train", "oof"), ("val", "val"), ("test", "test"))}
    return {s: np.load(p) for s, p in f.items()} if all(p.exists() for p in f.values()) else None
def csv(stem):
    f = {s: HERE / "cv_tf" / f"{stem}_{k}.csv" for s, k in (("train", "oof"), ("val", "val"), ("test", "test"))}
    return {s: pd.read_csv(p, dtype={"patient_id": str}).set_index("patient_id").reindex(ids[s])[codes].values for s, p in f.items()} if all(p.exists() for p in f.values()) else None
cands = {"lgb_clean": npy("clean", ""), "xgb_avg4": npy("v4", "avg_"), "xgb_bag5": npy("v4", "c_bag5_"), "l1_lam0.0003": npy("v4", "l1_lam0.0003_"), "l1_lam0.001": npy("v4", "l1_lam0.001_"), "l1_lam0.003": npy("v4", "l1_lam0.003_"),
         "stumps": npy("v4", "stumps_"), "pp_uniform": npy("pp", "person-period_uniform_"), "pp_x5": npy("pp", "person-period_official_x5_"),
         "pp_yearly": npy("pp2", "pp_yearly_uniform_"), "pp_yearly_decay": npy("pp2", "pp_yearly_decay_"), "pp1_uniform": npy("pp_mcw1", "person-period_uniform_"), "pp1_x5": npy("pp_mcw1", "person-period_official_x5_"),
         "pp_lgb": npy("pp_lgb", "person-period_uniform_"),
         "xgb_v5avg": npy("v5", "avg_"), "lgb_sweep_avg": npy("lgb_sweep", "avg_"), "lgb_nl8_mcs5": npy("lgb_sweep", "nl8_mcs5_"), "tf_hybrid": csv("hybrid")}
# auto-discovered candidates: level-2 stacking outputs, person-period sweep runs (ppds*/), offset-boosting runs
for p in sorted(glob.glob(str(HERE / "stack" / "*_oof.npy"))):
    stem = Path(p).name[:-8]
    if stem != "pdeath": cands["stack_" + stem] = npy("stack", stem + "_")
for dsdir in sorted(glob.glob(str(HERE / "ppds*"))):
    for p in sorted(glob.glob(str(Path(dsdir) / "*_oof.npy"))):
        stem = Path(p).name[:-8]; cands[f"{Path(dsdir).name}:{stem}"] = npy(Path(dsdir).name, stem + "_")
for p in sorted(glob.glob(str(HERE / "blocks" / "*_oof.npy"))):
    stem = Path(p).name[:-8]; cands["blocks_" + stem] = npy("blocks", stem + "_")
for p in sorted(glob.glob(str(HERE / "pplgb" / "*_oof.npy"))):
    stem = Path(p).name[:-8]; cands["pplgb:" + stem] = npy("pplgb", stem + "_")
for p in sorted(glob.glob(str(HERE / "offset" / "*_oof.npy"))):
    stem = Path(p).name[:-8]; cands["offset_" + stem] = npy("offset", stem + "_")
for sub in ("screen", "mix", "screen_g", "stack_g", "blocks_g", "v4_g", "gstack", "screen_gc", "stack_gc", "blocks_gc", "v4_gc", "screen_gcr", "stack_gcr", "blocks_gcr", "v4_gcr", "screen_gcr3", "blocks_gcr3", "stack_gcr3", "nn"):   # round 6: screening / mixtures; round 7: genomics-augmented (_g) families
    for p in sorted(glob.glob(str(HERE / sub / "*_oof.npy"))):
        stem = Path(p).name[:-8]
        if stem != "pdeath": cands[f"{sub}:{stem}"] = npy(sub, stem + "_")
import sys, argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--exclude", default=""); _ap.add_argument("--siblings", action="store_true"); _ap.add_argument("--rank", action="store_true"); _ap.add_argument("--steps", type=int, default=8); _ap.add_argument("--tag", default=""); _ap.add_argument("--no-write", action="store_true"); _ap.add_argument("--cf-reps", type=int, default=1); _ap.add_argument("--bagged", action="store_true"); _ap.add_argument("--only", default=""); _ap.add_argument("--shrink", type=float, default=0.0); _ap.add_argument("--shrink-level", default="group")
ARGS = _ap.parse_args(); EXCLUDE = set(ARGS.exclude.split(",")) if ARGS.exclude else set()
cands = {k: v for k, v in cands.items() if v is not None and k not in EXCLUDE and not any(x in k for x in EXCLUDE)}
if ARGS.only: cands = {k: v for k, v in cands.items() if k in set(ARGS.only.split(","))}   # exact names (substring matching silently pulled in e.g. ppds:d3_all_f0.* for ppds:d3_all)
print("candidates:", list(cands))
nonterm_only = {"pp_uniform", "pp_x5", "pp_yearly", "pp_yearly_decay", "pp1_uniform", "pp1_x5", "pp_lgb"} | {k for k in cands if "_nonterm" in k}
def ap(y, p): return average_precision_score(y, p) if y.sum() else 0.0

def greedy(A, y, steps=8):
    """A: n x K candidate probs; returns weights (counts/steps) maximizing AP with replacement."""
    K = A.shape[1]; sel = []; cur = np.zeros(len(y)); best = -1
    for _ in range(steps):
        scores = [ap(y, (cur + A[:, k]) / (len(sel) + 1)) for k in range(K)]; k = int(np.argmax(scores))
        if scores[k] < best - 1e-9 and sel: break
        best = scores[k]; sel.append(k); cur = cur + A[:, k]
    w = np.bincount(sel, minlength=K) / len(sel); return w

rng = np.random.default_rng(0); n = len(ids["train"]); HALVES = [rng.random(n) < 0.5 for _ in range(ARGS.cf_reps)]
names = list(cands)
oof_cf = np.zeros((n, 40)); pv = np.zeros((len(ids["val"]), 40)); pt = np.zeros((len(ids["test"]), 40))
MODULES = [["126906006", "92691004", "314994000"], ["162573006", "424132000", "254637007"], ["68496003", "713197008"], ["26929004", "22325002", "723857007"],
           ["53741008", "22298006", "88805009", "49436004"], ["64859006", "443165006", "65966004"], ["72892002", "156073000"], ["444814009", "36971009", "75498004"], ["195662009", "43878008"]]
SIB = {c: [x for g in MODULES if c in g for x in g if x != c] for c in codes}
def rk(a):
    """per-column rank transform to (0,1] (ties averaged) so that families with different calibration mix on equal footing"""
    from scipy.stats import rankdata
    return np.column_stack([rankdata(a[:, i]) / len(a) for i in range(a.shape[1])]) if ARGS.rank else a
W = np.zeros((40, len(cands) + (max(len(v) for v in SIB.values()) * 2 if ARGS.siblings else 0)))
STORE = {}
for j, c in enumerate(codes):
    use = [k for k in names if not (k in nonterm_only and c in TERMINAL)]
    A = {s: np.column_stack([np.where(Pv[s][:, j], 0, cands[k][s][:, j]) for k in use]) for s in ("train", "val", "test")}
    if ARGS.siblings and SIB[c]:   # sibling targets' lgb_clean + best pp predictions (same Synthea module -> shared hazard) as extra columns
        for sb in SIB[c]:
            jj = codes.index(sb); fams = ["lgb_sweep_avg"] + (["pp_yearly"] if sb not in TERMINAL else ["xgb_v5avg"])
            for f in fams:
                if f in cands:
                    for s in ("train", "val", "test"): A[s] = np.column_stack([A[s], np.where(Pv[s][:, j], 0, cands[f][s][:, jj])])
                    use = use + [f"sib:{sb}:{f}"]
    if ARGS.rank:
        # rank within each split separately is not comparable across splits; rank train for selection, and map val/test through
        # the train empirical CDF per column so that the final blend is a monotone transform of each family's probabilities
        Atr = A["train"]; cdf = lambda col, x: np.searchsorted(np.sort(col), x, side="right") / len(col)
        A = {"train": rk(Atr), "val": np.column_stack([cdf(Atr[:, i], A["val"][:, i]) for i in range(Atr.shape[1])]), "test": np.column_stack([cdf(Atr[:, i], A["test"][:, i]) for i in range(Atr.shape[1])])}
        for s in ("train", "val", "test"): A[s] = np.where(Pv[s][:, [j]], 0, A[s])
    y = Yv["train"][:, j]
    # cross-fitted OOF estimate (averaged over --cf-reps random half-splits)
    wbag = []
    for halves in HALVES:
        for h in (halves, ~halves):
            w = greedy(A["train"][h], y[h], ARGS.steps); wbag.append(w)
            if ARGS.shrink == 0: oof_cf[~h, j] += (A["train"][~h] @ w) / len(HALVES)
    # final weights on all OOF rows -> val/test
    w = greedy(A["train"], y, ARGS.steps)
    if ARGS.bagged: w = np.mean(wbag, 0)          # bagged ensemble selection (Caruana et al. 2004): average of the half-sample selections -> same predictor the cross-fitted OOF evaluates
    if ARGS.shrink > 0:
        STORE[j] = dict(A=A, use=use, w_half=wbag, w_full=w); continue
    pv[:, j] = A["val"] @ w; pt[:, j] = A["test"] @ w
    for k, wk in zip(use, w):
        if k in names: W[j, names.index(k)] = wk
if ARGS.shrink > 0:
    # hierarchical ensemble weights: per-target greedy weights shrunk toward the mean weights of the target's group
    # (acc / chain / term, or the Synthea module), evaluated with the same half-split cross-fitting as the plain version
    ACC_ = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
    if ARGS.shrink_level == "module": GRP = {j: next((tuple(g) for g in MODULES if c in g), ("solo", c)) for j, c in enumerate(codes)}
    else: GRP = {j: ("term" if c in TERMINAL else "acc" if c in ACC_ else "chain") for j, c in enumerate(codes)}
    def to_global(w, use):
        g = np.zeros(len(names))
        for k, wk in zip(use, w):
            if k in names: g[names.index(k)] = wk
        return g
    def from_global(g, use):
        v = np.array([g[names.index(k)] if k in names else 0.0 for k in use]); return v / v.sum() if v.sum() > 0 else v
    def blend(wj, gm, use):
        w = (1 - ARGS.shrink) * wj + ARGS.shrink * from_global(gm, use); return w / w.sum() if w.sum() > 0 else wj
    HL = [h for halves in HALVES for h in (halves, ~halves)]
    for hi, h in enumerate(HL):
        Gm = {}
        for j, st in STORE.items(): Gm.setdefault(GRP[j], []).append(to_global(st["w_half"][hi], st["use"]))
        Gm = {g: np.mean(v, 0) for g, v in Gm.items()}
        for j, st in STORE.items():
            w = blend(st["w_half"][hi], Gm[GRP[j]], st["use"]); oof_cf[~h, j] += (st["A"]["train"][~h] @ w) / len(HALVES)
    Gm = {}
    for j, st in STORE.items(): Gm.setdefault(GRP[j], []).append(to_global(st["w_full"], st["use"]))
    Gm = {g: np.mean(v, 0) for g, v in Gm.items()}
    for j, st in STORE.items():
        w = blend(st["w_full"], Gm[GRP[j]], st["use"]); pv[:, j] = st["A"]["val"] @ w; pt[:, j] = st["A"]["test"] @ w
        for k, wk in zip(st["use"], w):
            if k in names: W[j, names.index(k)] = wk
def group_avg(p, Pm):
    p = p.copy()
    for grp in GROUPS:
        jj = [codes.index(x) for x in grp]; m = p[:, jj].mean(1, keepdims=True); p[:, jj] = np.where(Pm[:, jj], 0, m)
    return p
def mAP(p, s): return float(np.mean([ap(Yv[s][:, j], np.where(Pv[s][:, j], 0, p[:, j])) for j in range(40)]))
print("\nsingle candidates (OOF mAP / val mAP):")
for k in names:
    o = cands[k]["train"].copy(); v = cands[k]["val"].copy()
    if k in nonterm_only:
        for c in TERMINAL: o[:, codes.index(c)] = cands["lgb_clean"]["train"][:, codes.index(c)]; v[:, codes.index(c)] = cands["lgb_clean"]["val"][:, codes.index(c)]
    print(f"  {k:12s} OOF={mAP(o,'train'):.4f}  val={mAP(v,'val'):.4f}" + ("  (terminal targets filled from lgb_clean)" if k in nonterm_only else ""))
fv, ft = group_avg(np.where(Pv["val"], 0, pv), Pv["val"]), group_avg(np.where(Pv["test"], 0, pt), Pv["test"])
print(f"\ngreedy ensemble: cross-fitted OOF mAP={mAP(group_avg(oof_cf, Pv['train']),'train'):.4f} | val mAP={mAP(fv,'val'):.4f}  val macroAUROC={np.mean([roc_auc_score(Yv['val'][:, j], fv[:, j]) for j in range(40)]):.4f}")
print("average weight per candidate:", dict(zip(names, W[:, :len(names)].mean(0).round(3))))
if ARGS.no_write: sys.exit(0)
pd.DataFrame(W[:, :len(names)], index=[desc[c][:30] for c in codes], columns=names).round(2).to_csv(OUT / f"weights{ARGS.tag}.csv")
pd.DataFrame(ft, index=pd.Index(ids["test"], name="patient_id"), columns=codes).to_csv(ROOT / f"predictions_final{ARGS.tag}.csv")
pd.DataFrame(fv, index=pd.Index(ids["val"], name="patient_id"), columns=codes).to_csv(OUT / f"val_predictions_final{ARGS.tag}.csv")
json.dump({"candidates": names, "oof_crossfit_mAP": mAP(group_avg(oof_cf, Pv["train"]), "train"), "val_mAP": mAP(fv, "val"), "avg_weights": dict(zip(names, W.mean(0).round(3).tolist()))}, open(OUT / f"summary{ARGS.tag}.json", "w"), indent=1)
np.save(OUT / f"oof_crossfit{ARGS.tag}.npy", group_avg(oof_cf, Pv["train"]))
ACC = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
gi = {"acc": [codes.index(c) for c in ACC], "term": [codes.index(c) for c in TERMINAL]}; gi["chain"] = [j for j in range(40) if j not in gi["acc"] and j not in gi["term"]]
a_cf = np.array([ap(Yv["train"][:, j], np.where(Pv["train"][:, j], 0, group_avg(oof_cf, Pv["train"])[:, j])) for j in range(40)]); a_ref = np.array([ap(Yv["train"][:, j], np.where(Pv["train"][:, j], 0, cands["lgb_clean"]["train"][:, j])) for j in range(40)]) if "lgb_clean" in cands else a_cf
print("by group (cross-fitted OOF vs lgb_clean):", {g: (round(float(a_cf[idx].mean()), 4), round(float(a_ref[idx].mean()), 4)) for g, idx in gi.items()})
pd.DataFrame({"cond": [desc[c][:32] for c in codes], "oof_crossfit": a_cf.round(3), "lgb_clean": a_ref.round(3), "gain": (a_cf - a_ref).round(3)}, index=codes).sort_values("gain").to_csv(OUT / f"per_condition_final{ARGS.tag}.csv")
print("wrote predictions_final.csv")
if os.environ.get("EXPORT_DIR"):
    E = Path(os.environ["EXPORT_DIR"]) / "assembly"; E.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(W[:, :len(names)], index=codes, columns=names).to_csv(E / "weights.csv")
    for k_ in names:
        for s, kk in (("train", "oof"), ("val", "val"), ("test", "test")): np.save(E / f"{k_.replace(':', '__')}_{kk}.npy", cands[k_][s])
    json.dump({"candidates": names, "codes": codes, "ids": {s: ids[s].tolist() for s in ids}, "rank": bool(ARGS.rank), "steps": ARGS.steps, "groups": GROUPS, "terminal": TERMINAL, "oof_crossfit_mAP": mAP(group_avg(oof_cf, Pv["train"]), "train"), "val_mAP": mAP(fv, "val")}, open(E / "assembly_meta.json", "w"), indent=1)
    print("exported assembly ->", E)
if os.environ.get("WANDB_PROJECT"):
    import wandb; wandb.init(project=os.environ["WANDB_PROJECT"], name="assembly_rank_greedy", config=dict(stage="assembly", candidates=names, rank=bool(ARGS.rank), steps=ARGS.steps, cf_reps=ARGS.cf_reps))
    wandb.summary.update({"oof_crossfit_mAP": mAP(group_avg(oof_cf, Pv["train"]), "train"), "val_mAP": mAP(fv, "val"), "val_macroAUROC": float(np.mean([roc_auc_score(Yv["val"][:, j], fv[:, j]) for j in range(40)])), **{f"oof_{g}": float(a_cf[idx].mean()) for g, idx in gi.items()}})
    wandb.log({"per_target": wandb.Table(columns=["code", "cond", "oof_crossfit_AP"] + names, data=[[c, desc[c][:40], float(a_cf[j])] + [float(W[j, i]) for i in range(len(names))] for j, c in enumerate(codes)])}); wandb.finish()
