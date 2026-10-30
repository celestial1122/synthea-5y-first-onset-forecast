"""Self-consistency / lower-bound test: is the realized OOF AP what the model's own calibrated risks predict?
For each target: cross-validated isotonic calibration of the OOF score -> p_hat; simulate labels y~Bernoulli(p_hat) 300 times
and compute the expected AP of the *same ranking*. If realized AP ~ expected AP, the labels behave like coin flips with the
model's risks: the low AP is the small spread of true risks, not a mis-ranking. Because calibrated predictions are less
dispersed than the true risks (E[p_true | p_hat] = p_hat), E[AP | p_hat] is a lower bound on the Bayes-optimal AP.
Also reports the closed-form AP of a two-point risk mixture to show how AP scales with risk spread."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / "analysis"; DATA = ROOT / "research_intern_coding_test"
pats = pd.read_csv(DATA / "train_val/patients.csv", dtype=str, usecols=["Id"]).rename(columns={"Id": "PATIENT"})
spl = pd.read_csv(DATA / "patient_splits.csv", dtype=str).rename(columns={"Id": "PATIENT"})
ids = pats.merge(spl, on="PATIENT", how="inner").query("split == 'train'").PATIENT.values
M = np.load(HERE / "ppds/meta.npz", allow_pickle=True); off = (M["k"] == 0) & (M["split"] == "train")
pos = pd.Series(np.arange(off.sum()), index=M["pid"][off].astype(str)).reindex(ids).values; assert not np.isnan(pos).any()
Y = M["y"][off][pos]; P = M["prior"][off][pos].astype(bool); codes = list(M["codes"])
targets = pd.read_csv(DATA / "target_conditions.csv", dtype=str); desc = dict(zip(targets.CODE, targets.DESCRIPTION))
TERMINAL = ["88805009", "22298006", "233604007", "126906006", "92691004", "162573006", "424132000", "254637007", "314994000", "713197008"]
ACCIDENT = ["444814009", "195662009", "10509002", "44465007", "43878008", "36971009", "75498004", "39848009", "263102004", "62106007", "58150001", "16114001", "70704007", "283371005", "284551006"]
grp = np.array(["term" if c in TERMINAL else "acc" if c in ACCIDENT else "chain" for c in codes])
rng = np.random.RandomState(0); rows = []
for name, f in [("ensemble", HERE / "final/oof_crossfit.npy"), ("lgb_clean", HERE / "clean/oof.npy")]:
    S = np.load(f); assert S.shape == Y.shape
    for j, c in enumerate(codes):
        at = ~P[:, j]; s = S[at, j]; y = Y[at, j]
        if y.sum() < 5: continue
        # cross-validated calibration score -> probability: isotonic (sharp, over-disperses with 30-150 positives) and Platt (smooth)
        p_iso = np.zeros(len(s)); p_pl = np.zeros(len(s)); ls = np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))[:, None]
        for a, b in KFold(5, shuffle=True, random_state=1).split(s):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(s[a], y[a]); p_iso[b] = iso.predict(s[b])
            lr = LogisticRegression(C=100.0, max_iter=1000).fit(ls[a], y[a]); p_pl[b] = lr.predict_proba(ls[b])[:, 1]
        real = average_precision_score(y, s); real_iso = average_precision_score(y, p_iso); real_pl = average_precision_score(y, p_pl)
        sim_iso = np.array([average_precision_score(rng.binomial(1, p_iso), p_iso) for _ in range(300)])
        sim_pl = np.array([average_precision_score(rng.binomial(1, p_pl), p_pl) for _ in range(300)])
        rows.append(dict(model=name, code=c, cond=desc[c][:30], grp=grp[j], n_atrisk=int(at.sum()), pos=int(y.sum()), prev=y.mean(), sd_iso=p_iso.std(), sd_platt=p_pl.std(),
                         ap_real=real, ap_real_iso=real_iso, ap_real_platt=real_pl, exp_iso=sim_iso.mean(), exp_platt=sim_pl.mean(), exp_platt_lo=np.percentile(sim_pl, 5), exp_platt_hi=np.percentile(sim_pl, 95)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df[df.model == "ensemble"].sort_values(["grp", "prev"]).round(3).to_string(index=False))
for name in ("ensemble", "lgb_clean"):
    d = df[df.model == name]
    print(f"\n{name}: realized mAP raw {d.ap_real.mean():.4f} | Platt-calibrated ranking {d.ap_real_platt.mean():.4f} vs expected-under-Platt-risks {d.exp_platt.mean():.4f} (5-95% {d.exp_platt_lo.mean():.4f}-{d.exp_platt_hi.mean():.4f}) | isotonic ranking {d.ap_real_iso.mean():.4f} vs expected {d.exp_iso.mean():.4f}")
    print("  by group real_platt/exp_platt: " + ", ".join(f"{g} {d[d.grp==g].ap_real_platt.mean():.3f}/{d[d.grp==g].exp_platt.mean():.3f}" for g in ("acc", "chain", "term")))
    print(f"  targets with realized (Platt) AP above the 95th pct of expected: {(d.ap_real_platt > d.exp_platt_hi).sum()}, below the 5th pct: {(d.ap_real_platt < d.exp_platt_lo).sum()} of {len(d)}")
df.to_csv(HERE / "oracle" / "ap_consistency.csv", index=False)
# closed-form illustration: a two-point risk mixture (fraction q at risk r_hi, rest at r_lo) with prevalence pi
def exp_ap(p, n=20000, reps=200):
    return np.mean([average_precision_score(rng.binomial(1, p), p + 1e-9 * rng.rand(len(p))) for _ in range(reps)])
print("\nAP of a perfectly-known two-point risk model, prevalence 5%, n=20000 (how much risk spread it takes to lift AP):")
for q, lift in [(1.0, 1), (0.5, 1.5), (0.2, 3), (0.1, 5), (0.05, 10), (0.05, 15)]:
    pi = 0.05; r_hi = min(pi * lift, 0.95); r_lo = max((pi - q * r_hi) / (1 - q), 0.0) if q < 1 else pi
    p = np.where(rng.rand(20000) < q, r_hi, r_lo); print(f"  {q*100:4.0f}% of patients at {lift:>2}x prevalence risk ({r_hi:.2f}) , rest at {r_lo:.3f}: expected AP {exp_ap(p):.3f}")
