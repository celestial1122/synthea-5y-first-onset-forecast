"""Train-only initial-gradient and fixed-scale PL diagnostics; no fitting.

The original study, temperatures, lambdas and checkpoints remain immutable.
Default device is CPU. CUDA must be explicitly requested by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ranking_study.objectives import (macro_bce_logits, rank_loss,
    sample_plackett_luce, permutation_ap_rewards)

STUDY = ROOT / "artifacts/group_ranking"
SEEDS = (17, 42, 73)
METHODS = ("bce", "ranknet", "listnet", "smoothap", "rloo_ap", "grpo_ap")
RL = {"rloo_ap", "grpo_ap"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std_population": float(values.std()),
            "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def gradient_statistics(gradients, bce_gradient, frozen_lambda):
    """Finite-draw descriptions; covariance trace is ddof=1, not per-coordinate mean."""
    values = torch.stack(gradients).detach().cpu().to(torch.float64)
    reference = bce_gradient.detach().cpu().to(torch.float64)
    n = values.shape[0]
    mean = values.mean(dim=0)
    mean_norm = float(mean.norm())
    bce_norm = float(reference.norm())
    norms = values.norm(dim=1)
    cosine = float(torch.dot(mean, reference) / (mean.norm() * reference.norm())) if mean_norm and bce_norm else None
    result = {"n_gradient_draws": n, "parameter_dimension": int(values.shape[1]),
              "norm_of_mean_gradient": mean_norm, "mean_of_draw_gradient_norms": float(norms.mean()),
              "draw_gradient_norms": vector_summary(norms.numpy()),
              "rms_draw_gradient_norm": float(values.square().sum(dim=1).mean().sqrt()),
              "cosine_mean_gradient_with_BCE": cosine, "diagnostic_BCE_gradient_norm": bce_norm,
              "frozen_lambda": frozen_lambda,
              "lambda_times_mean_gradient_norm_over_BCE_norm": frozen_lambda * mean_norm / bce_norm if bce_norm else None,
              "lambda_times_mean_draw_norm_over_BCE_norm": frozen_lambda * float(norms.mean()) / bce_norm if bce_norm else None}
    if n > 1:
        trace_cov = float((values - mean).square().sum() / (n - 1))
        signal_squared = mean_norm ** 2 - trace_cov / n
        noise = math.sqrt(trace_cov)
        result.update({"sample_gradient_covariance_trace_ddof1": trace_cov,
            "estimated_noise_norm_of_one_draw": noise,
            "estimated_MC_standard_error_norm_of_mean": noise / math.sqrt(n),
            "naive_signal_to_single_draw_noise_ratio": mean_norm / noise if noise else None,
            "naive_signal_to_mean_estimator_noise_ratio": mean_norm / (noise / math.sqrt(n)) if noise else None,
            "noise_corrected_signal_norm_squared_unclipped": signal_squared,
            "noise_corrected_signal_norm_clipped": math.sqrt(max(0., signal_squared)),
            "noise_corrected_signal_to_single_draw_noise_ratio": math.sqrt(max(0., signal_squared)) / noise if noise else None,
            "lambda_squared_times_covariance_trace": frozen_lambda ** 2 * trace_cov,
            "finite_draw_caution": "||mean g||^2 includes trace(covariance)/D sampling noise. The unbiased signal-squared plug-in can be negative; clipping to0 is only display convenience. SNR/cosine are descriptive, not confidence intervals or proof of an identifiable population gradient."})
        directional = values @ reference / reference.norm() if bce_norm else None
        if directional is not None:
            result["gradient_projection_onto_BCE_unit_direction"] = {
                "mean": float(directional.mean()), "sample_std": float(directional.std(correction=1)),
                "standard_error_of_mean": float(directional.std(correction=1) / math.sqrt(n))}
    else:
        result.update({"sample_gradient_covariance_trace_ddof1": None,
                       "conditional_sampling_covariance_trace": 0.,
                       "noise_note": "Deterministic objective, evaluated once; no stochastic covariance estimated."})
    return result


def verify_inputs():
    manifest = read(STUDY / "frozen_manifest.json")
    require(manifest["specification"]["seeds"] == list(SEEDS), "Frozen seeds changed")
    require(manifest["specification"]["methods"] == list(METHODS), "Frozen methods changed")
    for key in ("source_sha256", "feature_and_reference_sha256"):
        for name, expected in manifest[key].items():
            require(sha(ROOT / name) == expected, f"Frozen file changed: {name}")
    require(sha(ROOT / manifest["previous_manifest_path"]) == manifest["previous_manifest_sha256"],
            "Previous study manifest changed")
    run_records = {}
    for seed in SEEDS:
        for method in METHODS:
            folder = STUDY / "runs" / f"{method}-seed{seed}"
            integrity = read(folder / "artifact_sha256.json")
            for name in ("initialization.json", "config.json", "history.jsonl", "best_head.pt", "initial_head.pt"):
                require(name in integrity and sha(folder / name) == integrity[name], f"Run evidence changed: {folder / name}")
            run_records[f"{method}-seed{seed}"] = {"folder": folder,
                "initialization": read(folder / "initialization.json"), "config": read(folder / "config.json"),
                "checked_artifact_sha256": {name: integrity[name] for name in ("initialization.json", "config.json", "history.jsonl", "best_head.pt", "initial_head.pt")}}
    return manifest, run_records


def gradient(loss, parameters):
    pieces = torch.autograd.grad(loss, parameters)
    require(all(bool(torch.isfinite(value).all()) for value in pieces), "Nonfinite diagnostic gradient")
    return torch.cat([value.detach().reshape(-1).cpu().to(torch.float64) for value in pieces])


def train_history(folder):
    # Files also contain validation fields, which are not selected or used here.
    rows = [{key: value for key, value in json.loads(line).items()
             if key == "epoch" or key.startswith(("train/", "rank/"))}
            for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(rows and rows[0]["epoch"] == 0, "Missing training epoch0")
    active = [row for row in rows if row["epoch"] > 0]
    final = rows[-1]
    result = {"last_epoch": final["epoch"], "theta0_train_mAP": rows[0]["train/mAP"],
              "last_train_mAP": final["train/mAP"],
              "last_minus_theta0_train_mAP": final["train/mAP"] - rows[0]["train/mAP"],
              "max_observed_train_mAP": max(row["train/mAP"] for row in rows),
              "epoch10_train_mAP": next((row["train/mAP"] for row in rows if row["epoch"] == 10), None),
              "uses_validation_fields": False}
    for key in ("rank/reward_mean", "rank/reward_std", "rank/zero_var_fraction"):
        values = [row[key] for row in active if row.get(key) is not None]
        if values:
            result[key] = vector_summary(values)
    if "rank/zero_var_fraction" in result:
        result["mean_fraction_nonzero_variance_reward_groups"] = 1 - result["rank/zero_var_fraction"]["mean"]
    return result


def eligible_standard_deviations(logits, eligible):
    return torch.stack([logits[:, c][eligible[:, c]].to(torch.float64).std(correction=0)
                        for c in range(logits.shape[1])])


def scale_diagnostic(logits, labels, eligible, temperatures, codes, seed, samples):
    raw = logits.detach()
    reference_orders, greedy = [], []
    for c in range(raw.shape[1]):
        scores = raw[:, c][eligible[:, c]].cpu().to(torch.float64).numpy()
        truth = labels[:, c][eligible[:, c]].cpu().numpy()
        reference_orders.append(np.argsort(-scores, kind="stable"))
        greedy.append(float(average_precision_score(truth, scores)) if truth.any() else 0.)
    reports = []
    for scale in (.5, 1., 2.):
        generator = torch.Generator(device=raw.device).manual_seed(seed + 37041)
        rng_before = tensor_sha(generator.get_state())
        per_class, rewards_all, scaled_greedy = [], [], []
        with torch.no_grad():
            for c, code in enumerate(codes):
                scores = raw[:, c][eligible[:, c]]
                truth = labels[:, c][eligible[:, c]].bool()
                scaled_scores = scores * scale
                real = scaled_scores.cpu().to(torch.float64).numpy()
                require(np.array_equal(np.argsort(-real, kind="stable"), reference_orders[c]), "Positive scaling changed greedy order")
                ap = float(average_precision_score(truth.cpu().numpy(), real)) if bool(truth.any()) else 0.
                require(abs(ap - greedy[c]) <= 1e-12, "Positive scaling changed greedy AP")
                scaled_greedy.append(ap)
                if bool(truth.any()) and bool((~truth).any()):
                    permutations = sample_plackett_luce(scaled_scores / temperatures[c], samples, generator)
                    rewards = permutation_ap_rewards(truth, permutations, validate=False)
                else:
                    rewards = torch.full((samples,), ap, dtype=torch.float64, device=raw.device)
                rewards_all.append(rewards.cpu())
                per_class.append({"code": code, "greedy_AP_unchanged": ap,
                                  "sampled_AP_mean": float(rewards.mean().cpu()),
                                  "sampled_AP_std_population": float(rewards.std(correction=0).cpu())})
        reward_matrix = torch.stack(rewards_all)
        macro = reward_matrix.mean(dim=0)
        reports.append({"positive_score_scale": scale, "fixed_initial_temperature": True,
            "n_PL_samples_per_disease": samples, "common_Gumbel_seed": seed + 37041,
            "rng_before": rng_before, "rng_after": tensor_sha(generator.get_state()),
            "greedy_macro_AP_unchanged": float(np.mean(scaled_greedy)),
            "sampled_macro_AP_mean": float(macro.mean()),
            "sampled_macro_AP_std_population": float(macro.std(correction=0)),
            "mean_within_disease_reward_std": float(reward_matrix.std(dim=1, correction=0).mean()),
            "per_condition": per_class})
    require(len({row["rng_after"] for row in reports}) == 1, "Scale comparisons did not consume identical Gumbel streams")
    return {"interpretation": "Strictly positive .5/1/2 scaling preserves every greedy order/AP but changes PL sampling sharpness at fixed tau. Higher sampled reward without a greedy AP change is not model-ranking improvement. No scale was selected or trained.",
            "score_basis": "Eligible real logits; tie-aware AP, avoiding sigmoid saturation/rounding artifacts.",
            "common_noise": True, "results": reports,
            "sampled_reward_delta_scale2_minus1": reports[2]["sampled_macro_AP_mean"] - reports[1]["sampled_macro_AP_mean"],
            "sampled_reward_delta_scale_half_minus1": reports[0]["sampled_macro_AP_mean"] - reports[1]["sampled_macro_AP_mean"]}


def analyze_seed(seed, device, draws, scale_samples, run_records):
    directory = STUDY / "features" / f"seed{seed}"
    feature_manifest = read(directory / "manifest.json")
    with np.load(directory / "train.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    head = torch.load(directory / "initial_head.pt", map_location="cpu", weights_only=True)
    x = torch.from_numpy(arrays["features"]).to(device)
    y = torch.from_numpy(arrays["labels"]).to(device)
    eligible = ~torch.from_numpy(arrays["prior_conditions"].astype(bool)).to(device)
    require(x.shape == (2791, 133) and y.shape == eligible.shape == (2791, 40), "Unexpected frozen train shapes")
    require(not bool((y.bool() & ~eligible).any()), "Positive label overlaps prior condition")
    weight = head["weight"].clone().to(device).requires_grad_()
    bias = head["bias"].clone().to(device).requires_grad_()
    parameters = (weight, bias)
    base = run_records[f"bce-seed{seed}"]["initialization"]
    temperatures = torch.tensor(base["temperature"], dtype=x.dtype, device=device)
    require(temperatures.shape == (40,) and bool((temperatures > 0).all()), "Invalid frozen temperatures")
    bce_gradient = gradient(macro_bce_logits(F.linear(x, weight, bias), y, eligible), parameters)
    bce_norm = float(bce_gradient.norm())
    require(bce_norm > 0 and math.isfinite(bce_norm), "Invalid initial BCE gradient")
    result = {"seed": seed, "n_train_patients": 2791, "n_diseases": 40,
        "parameter_dimension": int(bce_gradient.numel()), "diagnostic_generator_seed": seed + 37041,
        "diagnostic_BCE_gradient_norm": bce_norm, "initial_temperature": base["temperature"],
        "cache_sha256": {name: sha(directory / name) for name in ("manifest.json", "train.npz", "initial_head.pt")},
        "methods": {}}
    initial_logits = F.linear(x, weight, bias).detach()
    initial_std = eligible_standard_deviations(initial_logits, eligible)
    for method in METHODS:
        record = run_records[f"{method}-seed{seed}"]
        init = record["initialization"]
        require(init["temperature"] == base["temperature"] and init["theta0_sha256"] == base["theta0_sha256"], "Initial temperature/head differs across methods")
        require(init["calibration_source_split"] == "train" and init["calibration_training_patients"] == 2791,
                "Calibration used an undeclared split")
        fixed_lambda = float(init["lambda"])
        historical_ratio = fixed_lambda * float(init["rank_gradient_norm"]) / float(init["bce_gradient_norm"])
        require(abs(historical_ratio - (0. if method == "bce" else .25)) <= 1e-10, "Frozen initial calibration ratio is not declared .25")
        generator = torch.Generator(device=device).manual_seed(seed + 37041)
        rng_before = tensor_sha(generator.get_state())
        gradients, stats = [], []
        if method == "bce":
            gradients = [bce_gradient]
        else:
            for _ in range(draws if method in RL else 1):
                loss, stat = rank_loss(F.linear(x, weight, bias), y, eligible, method, temperatures,
                                      generator=generator, group_size=8, chunk_size=128)
                gradients.append(gradient(loss, parameters))
                stats.append(stat)
        record_result = gradient_statistics(gradients, bce_gradient, fixed_lambda)
        record_result.update({"frozen_historical_calibration": {
            "bce_gradient_norm": init["bce_gradient_norm"], "rank_gradient_norm": init["rank_gradient_norm"],
            "calibration_draws": init["rank_gradient_draws"], "lambda": fixed_lambda,
            "lambda_times_historical_rank_norm_over_historical_BCE_norm": historical_ratio,
            "lambda_or_tau_recalibrated": False},
            "generator_before": rng_before, "generator_after": tensor_sha(generator.get_state()),
            "gradient_mean_sha256": tensor_sha(torch.stack(gradients).mean(dim=0)),
            "actual_training_history": train_history(record["folder"]),
            "checked_run_artifact_sha256": record["checked_artifact_sha256"]})
        if stats:
            for key in ("reward_mean", "reward_std", "zero_var_fraction"):
                observed = [stat[key] for stat in stats if stat.get(key) is not None]
                if observed:
                    record_result[f"sampled_groups_{key}"] = vector_summary(observed)
        best = torch.load(record["folder"] / "best_head.pt", map_location="cpu", weights_only=True)
        with torch.no_grad():
            best_logits = F.linear(x, best["weight"].to(device), best["bias"].to(device))
            best_std = eligible_standard_deviations(best_logits, eligible)
        ratios = [float(best_std[c] / initial_std[c]) if float(initial_std[c]) > 1e-12 else None for c in range(40)]
        valid_ratios = [ratio for ratio in ratios if ratio is not None]
        record_result["best_head_train_logit_scale"] = {
            "population_std_over_eligible_train_patients": True,
            "ratios": vector_summary(valid_ratios), "n_ratio_above_1_05": sum(ratio > 1.05 for ratio in valid_ratios),
            "n_ratio_above_1_2": sum(ratio > 1.2 for ratio in valid_ratios),
            "per_condition": [{"code": code, "initial_logit_std": float(initial_std[c]),
                               "best_logit_std": float(best_std[c]), "ratio": ratios[c]}
                              for c, code in enumerate(feature_manifest["target_codes"])],
            "qualification": "Best head was already chosen under the frozen validation rule; only training logits are examined here. A std ratio is not proof that updates were pure rescaling."}
        result["methods"][method] = record_result
    require(result["methods"]["rloo_ap"]["generator_after"] == result["methods"]["grpo_ap"]["generator_after"],
            "RLOO/GRPO diagnostics consumed different random streams")
    result["same_RLOO_GRPO_sampling_stream"] = True
    result["scale_only_PL_reward_diagnostic"] = scale_diagnostic(initial_logits, y, eligible, temperatures,
        feature_manifest["target_codes"], seed, scale_samples)
    return result


def self_test():
    reference = torch.tensor([1., 0.], dtype=torch.float64)
    stats = gradient_statistics([torch.tensor([1., 0.]), torch.tensor([3., 0.])], reference, .25)
    require(stats["norm_of_mean_gradient"] == 2 and stats["sample_gradient_covariance_trace_ddof1"] == 2,
            "Gradient-statistic self-test failed")
    require(stats["noise_corrected_signal_norm_squared_unclipped"] == 3, "Noise correction self-test failed")
    require(stats["cosine_mean_gradient_with_BCE"] == 1, "Cosine self-test failed")
    stats = gradient_statistics([torch.tensor([1., 0.]), torch.tensor([-1., 0.])], reference, .25)
    require(stats["cosine_mean_gradient_with_BCE"] is None and stats["noise_corrected_signal_norm_squared_unclipped"] == -1,
            "Zero-mean noise self-test failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--scale-samples", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--no-write", action="store_true", help="Smoke diagnostics only, never write a partial final report")
    args = parser.parse_args()
    require(args.draws >= 2 and args.scale_samples >= 2, "At least two gradient and scale samples required")
    require(len(set(args.seeds)) == len(args.seeds) and all(seed in SEEDS for seed in args.seeds), "Invalid/duplicate seeds")
    require(args.no_write or args.seeds == list(SEEDS), "Final report requires all declared seeds")
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "Explicit CUDA request unavailable; no silent fallback")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    self_test()
    manifest, run_records = verify_inputs()
    result = {"schema_version": 1, "purpose": "Frozen theta0 train-only gradient/noise and score-scale diagnostics; no fitting or parameter selection.",
        "device": str(device), "dtype": "float32 head/objective, float64 gradient statistics/rewards",
        "strict_deterministic": True, "n_RL_gradient_draws": args.draws, "policy_group_size": 8,
        "scale_samples_per_condition": args.scale_samples, "source_manifest_sha256": sha(STUDY / "frozen_manifest.json"),
        "frozen_source_sha256": manifest["source_sha256"], "analysis_script_sha256": sha(Path(__file__)),
        "scope_limits": ["No validation/test feature arrays are decoded or used for gradients, temperature, lambda, scale choice, or any parameter update. Their raw bytes may be hashed to verify the frozen cache.",
            "All examined heads/features were already fitted/selected under the original study. No optimization or hyperparameter change occurs here.",
            "Gradient cosines compare minimization losses; positive means locally aligned descent directions, negative means opposition at theta0.",
            "Finite-D empirical mean gradients, covariance/SNR and cosines do not identify a high-confidence population signal by themselves.",
            "RLOO estimates expected sampled AP gradients; normalized group-relative GRPO-inspired gradients are not claimed unbiased for that objective.",
            "Nonzero group reward variance shows a sampling signal exists; it does not imply useful held-out generalization.",
            "Positive scaling may improve a sampled policy reward while preserving every greedy AP; this cannot be presented as model ranking improvement."],
        "self_tests_passed": True, "seeds": {}}
    for seed in args.seeds:
        result["seeds"][str(seed)] = analyze_seed(seed, device, args.draws, args.scale_samples, run_records)
        print(json.dumps({"seed_finished": seed, "device": str(device), "draws": args.draws}), flush=True)
    # Recheck frozen files after computation; the new diagnostic is outside the freeze.
    verify_inputs()
    if not args.no_write:
        output = STUDY / "gradient_diagnostics.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"saved": str(output.relative_to(ROOT)), "seeds": args.seeds}), flush=True)
    else:
        print(json.dumps({"smoke_passed": True, "written": False, "seeds": args.seeds}), flush=True)


if __name__ == "__main__":
    main()
