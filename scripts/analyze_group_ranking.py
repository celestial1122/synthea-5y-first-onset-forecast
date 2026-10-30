"""Gate-controlled analysis of six frozen-encoder ranking objectives.

Metrics are computed per fitted seed before averaging. Shared patient draws
cover all methods, seeds and selection rules. No probabilities are ensembled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ehr_forecast.experiment_analysis import _bootstrap_interval, _score_orders, _weighted_macro_metrics
from ehr_forecast.metrics import evaluate_predictions
from ranking_study.objectives import macro_bce_logits
from ranking_study.trainer import file_sha, load_feature_cache, tensor_sha

METHODS = ("bce", "ranknet", "listnet", "smoothap", "rloo_ap", "grpo_ap")
SEEDS = (17, 42, 73)
METRICS = ("mAP", "macro_auroc", "micro_ap", "brier", "at_risk_bce", "macro_bce")
BOOT_METRICS = ("mAP", "macro_auroc", "brier", "at_risk_bce", "macro_bce")
REQUIRED = {"config.json", "provenance.json", "initialization.json", "initial_head.pt", "environment.json", "history.jsonl",
            "theta0_validation_metrics.json", "epoch10_validation_metrics.json", "epoch10_head.pt", "epoch10_val_predictions.csv",
            "best_head.pt", "best.pt", "validation_metrics.json", "group_validation_metrics.json", "summary.json",
            "train_predictions.csv", "val_predictions.csv", "test_predictions.csv", "artifact_sha256.json"}
NOTE = ("Single models; fixed fitted encoders/seeds and repeatedly used validation. Patient intervals do not include refitting or "
        "undo checkpoint/method selection. Between-seed SD is descriptive. Multiple methods/metrics/groups are unadjusted. "
        "Every arm can select epoch0, so nonnegative improvement over theta0 is not evidence of success. Main control is matched-budget BCE.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def rel(path, root=ROOT):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def summary(values):
    good = [float(v) for v in values if v is not None]
    return dict(mean=float(np.mean(good)) if good else None,
                sample_std=float(np.std(good, ddof=1)) if len(good) > 1 else None,
                min=min(good) if good else None, max=max(good) if good else None, n_defined=len(good))


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def read_csv(path, ids, codes):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    require(rows[0] == ["patient_id", *codes], f"CSV CODE order mismatch: {path}")
    require([r[0] for r in rows[1:]] == list(ids), f"CSV patient order mismatch: {path}")
    p = np.asarray([r[1:] for r in rows[1:]], dtype=np.float64)
    require(p.shape == (len(ids), len(codes)) and np.isfinite(p).all()
            and (p >= 0).all() and (p <= 1).all(), f"Invalid probability matrix: {path}")
    return p


def score(y, p, prior, codes):
    result = evaluate_predictions(y, p, codes)
    e = ~prior.astype(bool)
    q = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -(y * np.log(q) + (1 - y) * np.log1p(-q))
    result["at_risk_bce"] = float(loss[e].mean())
    result["macro_bce"] = float(np.mean((loss * e).sum(0) / np.maximum(e.sum(0), 1)))
    return result


def selection_from_history(rows, patience=15, epochs=100):
    require([r["epoch"] for r in rows] == list(range(len(rows))) and rows, "History must start at epoch0 and be contiguous")
    best, best_epoch, stale, stop = -math.inf, None, 0, None
    for row in rows:
        value, epoch = row["val/mAP"], row["epoch"]
        if value > best + 1e-6:
            best, best_epoch, stale = value, epoch, 0
        elif epoch:
            stale += 1
        if stale >= patience:
            stop = epoch
            break
    require(rows[-1]["epoch"] <= epochs, "History exceeded declared budget")
    require((stop == rows[-1]["epoch"]) if stop is not None else rows[-1]["epoch"] == epochs,
            "Run stopped before its declared patience/budget or continued after stopping")
    return dict(best_epoch=best_epoch, best_validation_mAP=best, last_epoch=rows[-1]["epoch"], no_update_selected=best_epoch == 0)


def artifact_ledger(directory):
    require(all((directory / n).is_file() for n in REQUIRED), f"Incomplete run: {directory}")
    ledger = read(directory / "artifact_sha256.json")
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != "artifact_sha256.json"}
    require(set(ledger) == actual and REQUIRED - {"artifact_sha256.json"} <= set(ledger), f"Incomplete artifact ledger: {directory}")
    for name, expected in ledger.items():
        require(file_sha(directory / name) == expected, f"Run artifact changed: {directory / name}")
    return ledger


def check_reproduction(study, frozen, root, *, skip=False, development=False):
    if skip:
        require(development and study.resolve() != (root / "artifacts/group_ranking").resolve(),
                "--skip-reproduction is restricted to a separate development study directory")
        return dict(passed=False, skipped_for_development=True)
    consolidated = read(study / "reproduction_gates.json")
    require(consolidated.get("passed") is True and consolidated.get("frozen_manifest_sha256") == file_sha(study / "frozen_manifest.json"),
            "Reproduction gate failed or belongs to a different freeze")
    if "gate_files" in consolidated:
        gates = [read(root / p) for p in consolidated["gate_files"]]
    else:
        gates = consolidated.get("runs", [])
    require(len(gates) == 2 and {(r.get("method"), r.get("seed")) for r in gates} == {("bce", 17), ("grpo_ap", 17)},
            "Reproduction must cover BCE and GRPO seed17 exactly once")
    checked = []
    for gate in gates:
        require(gate.get("passed") is True and gate.get("checks") and all(v is True for v in gate["checks"].values()), "A/A gate component failed")
        first, second = [root / gate[k] for k in ("baseline_run_dir", "repeat_run_dir")]
        require(first.resolve() == (study / "runs" / f"{gate['method']}-seed17").resolve()
                and second.resolve().is_relative_to(study.resolve()) and first.resolve() != second.resolve(), "Wrong A/A directories")
        artifact_ledger(first); artifact_ledger(second)
        for name, expected in gate["artifact_sha256"].items():
            require(file_sha(root / name) == expected, f"A/A evidence changed: {name}")
        history = lambda p: [{k: v for k, v in json.loads(line).items() if k != "elapsed_seconds"}
                             for line in (p / "history.jsonl").read_text().splitlines()]
        config = lambda p: {k: v for k, v in read(p / "config.json").items() if k != "output_dir"}
        require(history(first) == history(second) and config(first) == config(second), "A/A history/config does not reproduce")
        for name in ("initialization.json", "environment.json", "provenance.json"):
            require(read(first / name) == read(second / name), f"A/A JSON differs: {name}")
        for name in ("initial_head.pt", "best_head.pt", "epoch10_head.pt"):
            a, b = [torch.load(p / name, map_location="cpu", weights_only=True) for p in (first, second)]
            require(tensor_sha(a) == tensor_sha(b), f"A/A head differs: {name}")
        a,b=[torch.load(p/"best.pt",map_location="cpu",weights_only=True) for p in (first,second)]
        require(tensor_sha(a["model_state_dict"])==tensor_sha(b["model_state_dict"])
                and a["model_config"]==b["model_config"] and a["target_codes"]==b["target_codes"], "A/A exported model differs")
        for name in ("train_predictions.csv", "val_predictions.csv", "test_predictions.csv", "epoch10_val_predictions.csv"):
            require((first / name).read_bytes() == (second / name).read_bytes(), f"A/A predictions differ: {name}")
        omit = {"elapsed_seconds", "checkpoint_sha256"}
        require({k:v for k,v in read(first / "summary.json").items() if k not in omit} ==
                {k:v for k,v in read(second / "summary.json").items() if k not in omit}, "A/A summary differs")
        checked.append(dict(method=gate["method"], seed=17, passed=True, repeat_run_dir=gate["repeat_run_dir"]))
    return dict(passed=True, independently_rechecked=checked, consolidated_sha256=file_sha(study / "reproduction_gates.json"))


def verify_study(study, root):
    frozen = read(study / "frozen_manifest.json")
    require(frozen["specification"]["methods"] == list(METHODS) and frozen["specification"]["seeds"] == list(SEEDS), "Methods/seeds changed")
    require(file_sha(root / frozen["previous_manifest_path"]) == frozen["previous_manifest_sha256"], "Previous freeze changed")
    old = read(root / frozen["previous_manifest_path"])
    for paths in (frozen["source_sha256"], frozen["feature_and_reference_sha256"], old["source_sha256"]):
        for name, expected in paths.items():
            require(file_sha(root / name) == expected, f"Frozen source/cache/reference changed: {name}")
    for name, expected in old["prepared_sha256"].items():
        require(file_sha(root / "artifacts/optimization/prepared_l512" / name) == expected, f"Original prepared data changed: {name}")
    return frozen


def cache_reference(study, seed, root):
    directory = study / "features" / f"seed{seed}"
    manifest, reference, encoder, initial, arrays = load_feature_cache(directory, root)
    require(manifest["seed"] == seed and initial["weight"].shape == (40, 133), "Wrong feature cache seed/dimensions")
    require([len(arrays[s]["patient_ids"]) for s in ("train", "val", "test")] == [2791,365,358], "Official cohort sizes changed")
    val = arrays["val"]
    require(int(val["history_group_long"].sum()) == 133 and not (val["labels"].astype(bool) & val["prior_conditions"].astype(bool)).any(), "Validation groups/labels changed")
    x = torch.from_numpy(arrays["train"]["features"])
    y = torch.from_numpy(arrays["train"]["labels"])
    eligible = ~torch.from_numpy(arrays["train"]["prior_conditions"].astype(bool))
    w, b = initial["weight"].clone().requires_grad_(), initial["bias"].clone().requires_grad_()
    logits = F.linear(x, w, b)
    with torch.no_grad():
        tau = torch.stack([.5 * logits[eligible[:,j],j].std(unbiased=False) for j in range(40)]).clamp(.25,2).numpy()
    gradients = torch.autograd.grad(macro_bce_logits(logits, y, eligible), (w,b))
    norm = float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
    with torch.inference_mode():
        p = torch.sigmoid(F.linear(torch.from_numpy(val["features"]), initial["weight"], initial["bias"])).numpy()
        p[val["prior_conditions"].astype(bool)] = 0
    require(abs(evaluate_predictions(val["labels"],p,reference["target_codes"])["mAP"] - manifest["checks"]["val_initial_mAP"]) <= 1e-10, "Theta0 cache no longer reconstructs saved mAP")
    return dict(manifest=manifest, reference=reference, encoder=encoder, initial=initial, arrays=arrays, tau=tau,
                bce_gradient_norm=norm, theta0_probabilities=p, feature_manifest_sha256=file_sha(directory / "manifest.json"))


def load_run(study, method, seed, frozen, cache, root):
    directory = study / "runs" / f"{method}-seed{seed}"
    ledger = artifact_ledger(directory)
    config, provenance = read(directory / "config.json"), read(directory / "provenance.json")
    for key, expected in frozen["specification"].items():
        require(config.get(key) == expected, f"Training setting differs: {directory}/{key}")
    require(config["method"] == method and config["seed"] == seed and config["device"] == "cuda"
            and (root / config["output_dir"]).resolve() == directory.resolve()
            and (root / config["feature_dir"]).resolve() == (study / "features" / f"seed{seed}").resolve(), "Wrong run method/seed/directory")
    require(provenance["frozen_manifest_sha256"] == file_sha(study / "frozen_manifest.json")
            and provenance["source_sha256"] == frozen["source_sha256"]
            and provenance["feature_manifest_sha256"] == cache["feature_manifest_sha256"], "Run provenance differs")
    init, env, saved_summary = [read(directory / f"{n}.json") for n in ("initialization", "environment", "summary")]
    head0 = torch.load(directory / "initial_head.pt", map_location="cpu", weights_only=True)
    require(tensor_sha(head0) == init["theta0_sha256"] == tensor_sha(cache["initial"]), "Common initial head differs")
    train = cache["arrays"]["train"]
    eligible = ~train["prior_conditions"].astype(bool)
    require(init["feature_manifest_sha256"] == cache["feature_manifest_sha256"]
            and init["calibration_source_split"] == "train" and init["calibration_training_patients"] == len(train["patient_ids"])
            and init["calibration_eligible_counts"] == eligible.sum(0).tolist()
            and init["calibration_positive_counts"] == (train["labels"] * eligible).sum(0).astype(int).tolist()
            and init["calibration_train_cache_sha256"] == cache["manifest"]["files_sha256"]["train.npz"], "Calibration input is not frozen training pool")
    require(np.allclose(init["temperature"], cache["tau"], atol=1e-6, rtol=1e-6)
            and np.isclose(init["bce_gradient_norm"], cache["bce_gradient_norm"], atol=1e-7, rtol=1e-5), "Train-only tau/BCE-gradient does not independently reconstruct")
    require(init["training_generator_seed"] == seed+17041 and init["calibration_generator_seed"] == seed+27041
            and init["training_generator_before"] == init["training_generator_after_calibration"], "Calibration consumed training RNG or changed streams")
    expected_lambda = 0. if method == "bce" else config["kappa"] * init["bce_gradient_norm"] / init["rank_gradient_norm"]
    require(init["bce_gradient_norm"]>0 and (init["rank_gradient_norm"]>0 if method!="bce" else init["rank_gradient_norm"]==0)
            and init["temperature_recipe"]==config["temperature_recipe"]
            and math.isfinite(expected_lambda) and np.isclose(init["lambda"],expected_lambda,rtol=1e-10,atol=1e-12)
            and init["rank_gradient_draws"] == (0 if method == "bce" else 4 if method in {"rloo_ap","grpo_ap"} else 1), "Gradient scaling/draw count changed")
    require(env["deterministic_enabled"] is True and env["deterministic_warn_only"] is False
            and env["amp_enabled"] is False and env["dtype"] == "float32" and env["dropout"] is False
            and env["cuda_matmul_allow_tf32"] is False and env["cudnn_allow_tf32"] is False
            and env["float32_matmul_precision"] == "highest" and env["optimizer"] == "Adam"
            and env["cublas_workspace_config"]==":4096:8", "Run did not use declared strict numerical setting")
    rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    require(finite_tree(rows) and finite_tree(init) and all(r["lambda"] == init["lambda"] for r in rows), "Nonfinite diagnostics or changing lambda")
    chosen = selection_from_history(rows, config["patience"], config["epochs"])
    require(all(saved_summary[key] == value for key,value in chosen.items()), "Saved selection differs from replayed common rule")
    require(saved_summary["checkpoint_sha256"] == file_sha(directory / "best.pt")
            and saved_summary["frozen_backbone_export_verified"] is True and saved_summary["test_metrics"] is None, "Invalid exported summary")
    require(abs(rows[0]["val/mAP"]-cache["manifest"]["checks"]["val_initial_mAP"])<=1e-10
            and saved_summary["theta0_validation_mAP"]==rows[0]["val/mAP"], "Theta0 performance changed across methods")
    require(len(rows)>10 and rows[10]["epoch"]==10 and saved_summary["epoch10_validation_mAP"]==rows[10]["val/mAP"], "Missing common fixed-epoch10 comparison")
    exported = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    require(exported["target_codes"] == cache["reference"]["target_codes"] and exported["model_config"] == cache["reference"]["model_config"], "Export model/CODE specification changed")
    require(set(exported["model_state_dict"]) == set(cache["encoder"]["model_state_dict"]), "Exported state keys changed")
    require(exported["ranking_study"]["method"]==method and exported["ranking_study"]["head_epoch"]==chosen["best_epoch"]
            and exported["config"]==config, "Exported ranking metadata differs")
    for name, tensor in cache["encoder"]["model_state_dict"].items():
        if not name.startswith("classification_head."):
            require(torch.equal(tensor, exported["model_state_dict"][name]), f"Frozen encoder changed: {method}/{seed}/{name}")
    codes = cache["reference"]["target_codes"]
    predictions = {}
    for phase, head_file, csv_file, metrics_file in (("best","best_head.pt","val_predictions.csv","validation_metrics.json"),
            ("epoch10","epoch10_head.pt","epoch10_val_predictions.csv","epoch10_validation_metrics.json")):
        head = torch.load(directory / head_file, map_location="cpu", weights_only=True)
        require(set(head)=={"weight","bias"} and all(torch.isfinite(t).all() for t in head.values()), "Invalid fitted head")
        val = cache["arrays"]["val"]
        p = read_csv(directory / csv_file, val["patient_ids"].astype(str).tolist(), codes)
        require((p[val["prior_conditions"].astype(bool)]==0).all(), "Prior masking changed")
        with torch.inference_mode():
            reconstructed = torch.sigmoid(F.linear(torch.from_numpy(val["features"]),head["weight"],head["bias"])).numpy()
            reconstructed[val["prior_conditions"].astype(bool)]=0
        require(float(np.max(np.abs(reconstructed-p)))<=1e-6, "Feature head fails to reconstruct prediction CSV")
        metrics=score(val["labels"],p,val["prior_conditions"],codes)
        recorded=read(directory / metrics_file)
        require(all(abs(metrics[k]-recorded[k])<=1e-10 for k in METRICS), "Recomputed validation metrics differ")
        require(abs(metrics["mAP"]-rows[chosen["best_epoch"] if phase=="best" else 10]["val/mAP"])<=1e-10, "CSV/history score differs")
        if phase=="best":
            require(all(torch.equal(head[k],exported["model_state_dict"][f"classification_head.{k}"]) for k in ("weight","bias")), "Exported head differs")
            if chosen["no_update_selected"]:
                require(tensor_sha(head)==tensor_sha(cache["initial"]), "Epoch0 fallback changed the head")
        predictions[phase]=p
    for split in ("train","test"):
        arrays=cache["arrays"][split]
        p=read_csv(directory / f"{split}_predictions.csv",arrays["patient_ids"].astype(str).tolist(),codes)
        require((p[arrays["prior_conditions"].astype(bool)]==0).all(), "Prior mask differs in exported predictions")
    return dict(method=method,seed=seed,run_dir=rel(directory,root),summary=saved_summary,initialization=init,environment=env,history=rows,
                artifact_ledger_sha256=file_sha(directory / "artifact_sha256.json")), predictions


def shared_bootstrap(labels, prior, predictions, codes, masks, *, n_bootstrap=2000, seed=2026):
    """One patient draw jointly covers every phase/method/seed within a group."""
    results={}
    for group,mask in masks.items():
        y,e=labels[mask],~prior[mask].astype(bool)
        local={(phase,m,s):p[mask] for phase,methods in predictions.items() for m,seeds in methods.items() for s,p in seeds.items()}
        orders={k:_score_orders(p) for k,p in local.items()}
        losses={}
        points={}
        for key,p in local.items():
            q=np.clip(p,1e-7,1-1e-7)
            losses[key]=(-(y*np.log(q)+(1-y)*np.log1p(-q)))*e
            points[key]=score(y,p,prior[mask],codes)
        draws={(phase,m,k):[] for phase in predictions for m in METHODS[1:] for k in BOOT_METRICS}
        rng=np.random.default_rng(seed)
        valid=[]
        for draw in range(n_bootstrap):
            weights=np.bincount(rng.integers(0,len(y),size=len(y)),minlength=len(y)).astype(float)
            denom=weights@e
            values={}
            for key,p in local.items():
                row=_weighted_macro_metrics(y,orders[key],weights)
                row["brier"]=float(weights@np.mean((p-y)**2,axis=1)/weights.sum())
                cell_loss=weights@losses[key]
                row["at_risk_bce"]=float(cell_loss.sum()/denom.sum())
                row["macro_bce"]=float(np.mean(cell_loss/np.maximum(denom,1)))
                values[key]=row
            valid.append(values[next(iter(values))]["n_auroc_valid"])
            for phase in predictions:
                for method in METHODS[1:]:
                    for metric in BOOT_METRICS:
                        ds=[values[(phase,method,s)][metric]-values[(phase,"bce",s)][metric] for s in SEEDS
                            if values[(phase,"bce",s)][metric] is not None]
                        draws[(phase,method,metric)].append(float(np.mean(ds)) if ds else None)
            if n_bootstrap>=500 and (draw+1)%500==0:
                print(f"Paired bootstrap {group}: {draw+1}/{n_bootstrap} shared patient draws",flush=True)
        comparisons={}
        for phase in predictions:
            comparisons[phase]={}
            for method in METHODS[1:]:
                per_seed=[dict(seed=s,**{k:points[(phase,method,s)][k]-points[(phase,"bce",s)][k]
                            if points[(phase,"bce",s)][k] is not None else None for k in METRICS}) for s in SEEDS]
                means={k:summary([r[k] for r in per_seed]) for k in METRICS}
                comparisons[phase][method]=dict(per_seed_delta=per_seed,between_seed=means,
                    intervals={k:_bootstrap_interval(draws[(phase,method,k)],means[k]["mean"],.95) for k in BOOT_METRICS},
                    wins_mAP=sum(r["mAP"]>1e-12 for r in per_seed),ties_mAP=sum(abs(r["mAP"])<=1e-12 for r in per_seed))
        results[group]=dict(n_patients=int(mask.sum()),n_positive_by_target=y.sum(0).astype(int).tolist(),comparisons=comparisons,
                           n_bootstrap=n_bootstrap,bootstrap_seed=seed,confidence=.95,
                           bootstrap_n_auroc_valid=dict(min=min(valid),max=max(valid)),
                           resampling="Same patient multiplicity across all methods, fitted seeds and best/epoch10 phases; metrics before seed means.")
    return results


def ranking_diagnostics(labels, prior, predictions, codes, ids, descriptions):
    result={}
    for method in METHODS[1:]:
        contributions=[]; patients={}; old_cases={f"V{i+1:03d}":[] for i in (84,210,76,337)}
        for j,code in enumerate(codes):
            y=labels[:,j].astype(bool); npos=int(y.sum()); per_seed=[]
            for seed in SEEDS:
                a,b=predictions["bce"][seed][:,j],predictions[method][seed][:,j]
                aps=[evaluate_predictions(y[:,None],p[:,None],[code])["mAP"] for p in (a,b)]
                per_seed.append(dict(seed=seed,AP_before=aps[0],AP_after=aps[1],delta=aps[1]-aps[0]))
                reconstructed=0.
                for i in range(len(ids)):
                    alias=f"V{i+1:03d}"; states=[]
                    if not y[i] and alias not in old_cases:
                        continue
                    for p in (a,b):
                        greater,equal=p>p[i],p==p[i]
                        states.append(dict(probability=float(p[i]),rank_best=int(greater.sum())+1,rank_worst=int((greater|equal).sum()),
                                           negative_above=int((greater & ~y & ~prior[:,j].astype(bool)).sum()),
                                           threshold_precision=float(y[greater|equal].sum()/(greater|equal).sum())))
                    contribution=(states[1]["threshold_precision"]-states[0]["threshold_precision"])/npos/len(codes) if y[i] else None
                    if y[i]:
                        reconstructed+=contribution
                        patients.setdefault(ids[i],dict(patient_id=ids[i],alias=alias,mean_mAP_contribution=0.,target_contributions=[]))
                        patients[ids[i]]["mean_mAP_contribution"]+=contribution/len(SEEDS)
                        patients[ids[i]]["target_contributions"].append(dict(code=code,seed=seed,mAP_contribution=contribution))
                    if alias in old_cases:
                        old_cases[alias].append(dict(patient_id=ids[i],code=code,description=descriptions[j],seed=seed,
                            label=int(y[i]),prior=bool(prior[i,j]),bce=states[0],method=states[1],positive_mAP_contribution=contribution))
                require(abs(reconstructed-(aps[1]-aps[0])/len(codes))<1e-12, "Patient AP decomposition failed")
            contributions.append(dict(code=code,description=descriptions[j],n_positive=npos,per_seed=per_seed,
                mean_AP_delta=float(np.mean([r["delta"] for r in per_seed])),mean_mAP_contribution=float(np.mean([r["delta"] for r in per_seed]))/len(codes)))
        contributions.sort(key=lambda r:r["mean_mAP_contribution"])
        patient_rows=sorted(patients.values(),key=lambda r:r["mean_mAP_contribution"])
        result[method]=dict(comparator="matched-budget bce best vs method best",top_target_losses=contributions[:8],top_target_gains=list(reversed(contributions[-8:])),
                            all_target_contributions=contributions,top_patient_losses=patient_rows[:10],top_patient_gains=list(reversed(patient_rows[-10:])),old_cases=old_cases,
                            note="Algebraic threshold-precision attribution, not patient-intervention causation. Cases never enter training/calibration/reward selection.")
    return result


def plots(study, scores, groups, runs, *, suffix=""):
    caption=("Source: frozen feature/head runs; official validation n=365, long133/short232. Single-model metrics before seed means. "
             "Seed SD is not a confidence interval. Patient intervals condition on fitted models/selection. All arms may select epoch0; matched-budget BCE is the control.")
    def save(fig,name):
        fig.text(.01,.01,textwrap.fill(caption,width=210),fontsize=7,verticalalignment="bottom")
        fig.tight_layout(rect=(0,.09,1,.95)); fig.savefig(study/f"{name}{suffix}.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,group in zip(axes,("overall","long","short")):
        for i,method in enumerate(("theta0",*METHODS)):
            vals=[r["mAP"] for r in scores if r["group"]==group and r["method"]==method and r["phase"]==("theta0" if method=="theta0" else "best")]
            ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),fmt="o",capsize=3)
            ax.scatter([i-.06,i,i+.06],vals,s=12,alpha=.5)
        ax.set(title=f"{group}: single-model mAP mean ± seed SD",ylabel="mAP (0–1; higher is better)",xticks=range(7),xticklabels=("theta0",*METHODS))
        ax.tick_params(axis="x",rotation=35); ax.grid(axis="y",alpha=.2)
    save(fig,"mAP_by_group")
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,group in zip(axes,("overall","long","short")):
        for phase,offset,marker in (("best",-.08,"o"),("epoch10",.08,"s")):
            for i,method in enumerate(METHODS[1:]):
                ci=groups[group]["comparisons"][phase][method]["intervals"]["mAP"]
                color="C0" if phase=="best" else "C1"
                ax.hlines(i+offset,ci["ci_lower"],ci["ci_upper"],color=color)
                ax.plot(ci["estimate"],i+offset,marker,color=color,label=phase if i==0 else None)
        ax.axvline(0,color="black",lw=.8); ax.set(title=f"{group}: paired method minus BCE",xlabel="ΔmAP; conditional 95% patient interval",yticks=range(5),yticklabels=METHODS[1:]); ax.legend()
    save(fig,"paired_delta_mAP")
    for name,metric,label in (("gradient_trajectories","gradient_norm","Full-objective gradient L2 norm"),
                              ("selection_trajectories","val/mAP","Validation mAP (0–1)")):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,method in zip(axes.flat,METHODS):
            for seed in SEEDS:
                run=runs[f"{method}-seed{seed}"]; rows=[r for r in run["history"] if r.get(metric) is not None]
                ax.plot([r["epoch"] for r in rows],[r[metric] for r in rows],label=f"seed{seed}")
                if metric=="val/mAP":
                    ax.scatter(run["summary"]["best_epoch"],run["summary"]["best_validation_mAP"],s=20)
            ax.set(title=method,xlabel="Head epoch (updates)",ylabel=label); ax.grid(alpha=.2); ax.legend(fontsize=8)
        save(fig,name)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for row,method in zip(axes,("rloo_ap","grpo_ap")):
        for ax,key,label in zip(row,("rank/reward_mean","rank/reward_std","rank/zero_var_fraction"),("Training sampled AP reward mean","Within-group reward SD (AP units)","Zero-variance condition fraction")):
            for seed in SEEDS:
                values=[r for r in runs[f"{method}-seed{seed}"]["history"] if r.get(key) is not None]
                if values: ax.plot([r["epoch"] for r in values],[r[key] for r in values],label=f"seed{seed}")
            ax.set(title=method,xlabel="Head epoch (updates)",ylabel=label); ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig,"reward_trajectories")


def analyze(study, *, n_bootstrap=2000, skip_reproduction=False, development=False, root=ROOT):
    study,root=Path(study).resolve(),Path(root).resolve()
    require(n_bootstrap>=2 and (development or n_bootstrap==2000), "Formal study uses the predeclared 2000 bootstrap draws")
    frozen=verify_study(study,root)
    reproduction=check_reproduction(study,frozen,root,skip=skip_reproduction,development=development)
    caches={s:cache_reference(study,s,root) for s in SEEDS}
    base=caches[SEEDS[0]]; val=base["arrays"]["val"]; labels,prior=val["labels"],val["prior_conditions"].astype(bool)
    ids=val["patient_ids"].astype(str).tolist(); codes=base["reference"]["target_codes"]
    descriptions=read(root/"artifacts/optimization/prepared_l512/metadata.json")["target_descriptions"]
    for seed,cache in caches.items():
        for split in ("train","val","test"):
            for field in ("patient_ids","prior_conditions","history_group_long","full_history_length","anchor_dates"):
                require(np.array_equal(cache["arrays"][split][field],base["arrays"][split][field]), f"Feature seed metadata differs: {split}/{field}")
            if split!="test": require(np.array_equal(cache["arrays"][split]["labels"],base["arrays"][split]["labels"]), "Labels differ across feature seeds")
    predictions={phase:{m:{} for m in METHODS} for phase in ("best","epoch10")}; runs={}; scores=[]
    masks={"overall":np.ones(len(labels),bool),"long":val["history_group_long"].astype(bool),"short":~val["history_group_long"].astype(bool)}
    for method in METHODS:
        for seed in SEEDS:
            run,p=load_run(study,method,seed,frozen,caches[seed],root); runs[f"{method}-seed{seed}"]=run
            for phase in predictions:
                predictions[phase][method][seed]=p[phase]
                for group,mask in masks.items():
                    metrics=score(labels[mask],p[phase][mask],prior[mask],codes)
                    scores.append(dict(method=method,seed=seed,phase=phase,group=group,**{k:metrics[k] for k in METRICS}))
    first_environment=next(iter(runs.values()))["environment"]
    require(all(r["environment"]==first_environment for r in runs.values()), "Runtime/precision differs across methods/seeds")
    for seed in SEEDS:
        for group,mask in masks.items():
            metrics=score(labels[mask],caches[seed]["theta0_probabilities"][mask],prior[mask],codes)
            scores.append(dict(method="theta0",seed=seed,phase="theta0",group=group,**{k:metrics[k] for k in METRICS}))
    groups=shared_bootstrap(labels,prior,predictions,codes,masks,n_bootstrap=n_bootstrap)
    candidate=max(METHODS,key=lambda m:np.mean([r["mAP"] for r in scores if r["method"]==m and r["phase"]=="best" and r["group"]=="overall"]))
    if candidate=="bce":
        recommendation=dict(outcome="No ranking method exceeded matched-budget BCE in three-seed mean mAP.",mean_gain_over_BCE=0.,conditional_mAP_interval=None)
    else:
        effect=groups["overall"]["comparisons"]["best"][candidate]
        ci=effect["intervals"]["mAP"]
        recommendation=dict(outcome="Candidate mean exceeds BCE; conditional interval remains exploratory after method selection.",
                            mean_gain_over_BCE=effect["between_seed"]["mAP"]["mean"],conditional_mAP_interval=ci,
                            interval_strictly_positive=ci["ci_lower"]>0)
    result=dict(schema_version=1,controls_passed=True,uses_probability_ensemble=False,seeds=list(SEEDS),methods=list(METHODS),
        reproduction=reproduction,source_sha256=frozen["source_sha256"],frozen_manifest_sha256=file_sha(study/"frozen_manifest.json"),
        analysis_script_sha256=file_sha(Path(__file__)),cohort=dict(n_patients=len(ids),n_targets=len(codes),long=int(masks["long"].sum()),short=int(masks["short"].sum())),
        groups=groups,point_summaries={phase:{group:{m:{k:summary([r[k] for r in scores if r["phase"]==phase and r["group"]==group and r["method"]==m]) for k in METRICS}
            for m in (METHODS if phase!="theta0" else ("theta0",))} for group in masks} for phase in ("best","epoch10","theta0")},
        runs=runs,diagnostics=ranking_diagnostics(labels,prior,predictions["best"],codes,ids,descriptions),
        recommended_experimental_candidate=dict(method=candidate,selection="Highest three-seed mean overall validation mAP; fixed method order resolves exact ties.",
            demonstration_seed=42,demonstration_checkpoint=rel(study/"runs"/f"{candidate}-seed42"/"best.pt",root),
            candidate_is_ranking_method=candidate!="bce",automatic_promotion=False,root_predictions_changed=False,**recommendation),
        epoch0_selection=dict(included_for_every_arm=True,gain_over_theta0_is_success_evidence=False,
                             no_update_selected_by_method={m:sum(runs[f"{m}-seed{s}"]["summary"]["no_update_selected"] for s in SEEDS) for m in METHODS}),
        interpretation=[NOTE,"Theta0 gains are selection-constrained; compare every method against same-budget BCE and inspect epoch10 simultaneously.",
            "Candidate method is selected on these same validation data; its unadjusted interval cannot validate the selection itself.",
            "Long/short intervals are separate conditional analyses, not a formal history-group by objective interaction test.",
            "Training-only tau and BCE-gradient calibration are independently recomputed; rank-gradient magnitudes/streams/formula/source are checked, not GPU RNG replayed on CPU.",
            "Old case IDs are diagnostic only; no validation case enters objective construction/calibration or sampling."],
        development_without_reproduction=skip_reproduction)
    suffix=".development" if skip_reproduction else ""
    write(study/f"analysis{suffix}.json",result)
    with (study/f"scores{suffix}.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0])); writer.writeheader(); writer.writerows(scores)
    plots(study,scores,groups,runs,suffix=suffix)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir",type=Path,default=ROOT/"artifacts/group_ranking")
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--skip-reproduction",action="store_true")
    parser.add_argument("--development",action="store_true")
    args=parser.parse_args()
    result=analyze(args.study_dir,n_bootstrap=args.bootstrap,skip_reproduction=args.skip_reproduction,development=args.development)
    print(json.dumps(dict(passed=True,controls_passed=result["controls_passed"],candidate=result["recommended_experimental_candidate"],uses_probability_ensemble=False)))


if __name__=="__main__":
    main()
