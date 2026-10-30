"""Frozen-encoder, full-pool head training with matched selection/budgets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from ehr_forecast.metrics import evaluate_predictions
from ehr_forecast.submission import write_predictions
from .objectives import macro_bce_logits, rank_loss

METHODS = ("bce", "ranknet", "listnet", "smoothap", "rloo_ap", "grpo_ap")
RL_METHODS = {"rloo_ap", "grpo_ap"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def tensor_sha(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def load_feature_cache(directory, root):
    """Validate serialized provenance and arrays without touching the encoder."""
    directory, root = Path(directory), Path(root)
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("schema_version") == 1 and manifest["checks"]["passed"] is True,
            "Feature extraction checks did not pass")
    files = manifest["files_sha256"]
    required = {"train.npz", "val.npz", "test.npz", "initial_head.pt", "reference_checkpoint.json"}
    require(required.issubset(files), "Feature cache manifest is incomplete")
    for name, fingerprint in files.items():
        require(file_sha(directory / name) == fingerprint, f"Feature cache changed: {name}")
    reference = read_json(directory / "reference_checkpoint.json")
    reference_path = root / reference["path"]
    require(file_sha(reference_path) == reference["sha256"], "Reference checkpoint changed")
    head = torch.load(directory / "initial_head.pt", map_location="cpu", weights_only=True)
    require(set(head) == {"weight", "bias"}, "Initial head must contain only weight/bias")
    require(head["weight"].dtype == head["bias"].dtype == torch.float32, "Head must be float32")
    k, d = head["weight"].shape
    require(head["bias"].shape == (k,), "Initial head dimensions disagree")
    require(len(reference["target_codes"]) == k and reference["target_codes"] == manifest["target_codes"],
            "Target order differs between reference/cache")
    checkpoint = torch.load(reference_path, map_location="cpu", weights_only=True)
    for key in ("weight", "bias"):
        require(torch.equal(checkpoint["model_state_dict"][f"classification_head.{key}"], head[key]),
                "Feature initial head differs from the reference checkpoint")
    arrays = {}
    for split in ("train", "val", "test"):
        with np.load(directory / f"{split}.npz", allow_pickle=False) as archive:
            row = {name: archive[name] for name in archive.files}
        ids = row["patient_ids"].astype(str).tolist()
        n = len(ids)
        require(n > 0 and len(set(ids)) == n, f"Invalid {split} patient IDs")
        require(ids == manifest["splits"][split]["patient_ids"], f"Patient order differs: {split}")
        require(row["features"].dtype == np.float32 and row["features"].shape == (n, d)
                and np.isfinite(row["features"]).all(), f"Invalid float32 features: {split}")
        require(row["prior_conditions"].shape == (n, k), f"Invalid prior mask: {split}")
        require(np.isin(row["prior_conditions"], [0, 1]).all(), "Prior mask is not binary")
        if split == "test":
            require("labels" not in row, "Test labels must not be present")
        else:
            require(row["labels"].shape == (n, k) and np.isin(row["labels"], [0, 1]).all(),
                    f"Invalid labels: {split}")
            require(not np.any(row["labels"][row["prior_conditions"].astype(bool)]),
                    "Incident label overlaps known-prior disease")
        if "history_group_long" in row:
            require(row["history_group_long"].shape == (n,), "Invalid history groups")
        arrays[split] = row
    id_sets = [set(arrays[split]["patient_ids"].astype(str)) for split in ("train", "val", "test")]
    require(not any(id_sets[a] & id_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))), "Split patient overlap")
    return manifest, reference, checkpoint, head, arrays


def _head_state(head):
    return {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def _norm(gradients):
    return float(torch.stack([gradient.detach().float().square().sum() for gradient in gradients]).sum().sqrt())


def ridge_penalty(head, initial):
    # Average over diseases, including bias: no averaging over the feature axis.
    return .5 * ((head.weight - initial["weight"]).square().sum(dim=1)
                 + (head.bias - initial["bias"]).square()).mean()


@torch.inference_mode()
def evaluate_head(head, data, codes):
    head.eval()
    probabilities = torch.sigmoid(head(data["features"]).float())
    probabilities.masked_fill_(data["prior"], 0.)
    scores = probabilities.cpu().numpy()
    if "labels" not in data:
        return None, scores
    truth = data["labels"].cpu().numpy()
    eligible = (~data["prior"]).cpu().numpy()
    metrics = evaluate_predictions(truth, scores, codes)
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1 - 1e-7)
    losses = -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))
    metrics["at_risk_bce"] = float(losses[eligible].mean())
    metrics["macro_bce"] = float(np.mean((losses * eligible).sum(axis=0) / np.maximum(eligible.sum(axis=0), 1)))
    return metrics, scores


def _tensor_data(arrays, device):
    result = {"features": torch.from_numpy(arrays["features"]).to(device),
              "prior": torch.from_numpy(arrays["prior_conditions"].astype(bool)).to(device)}
    if "labels" in arrays:
        result["labels"] = torch.from_numpy(arrays["labels"].astype(np.float32)).to(device)
    return result


def calibrate(head, train_data, method, config, training_generator, calibration_generator):
    x, y, eligible = train_data["features"], train_data["labels"], ~train_data["prior"]
    with torch.no_grad():
        logits = head(x)
        tau = torch.stack([.5 * logits[eligible[:, k], k].std(unbiased=False)
                           if eligible[:, k].any() else logits.new_tensor(0.)
                           for k in range(logits.shape[1])]).clamp(.25, 2.).detach()
    parameters = tuple(head.parameters())
    bce = macro_bce_logits(head(x), y, eligible)
    bce_norm = _norm(torch.autograd.grad(bce, parameters))
    require(math.isfinite(bce_norm) and bce_norm > 1e-12, "Initial BCE gradient is nonfinite or zero")
    train_before = tensor_sha({"state": training_generator.get_state()})
    calibration_before = tensor_sha({"state": calibration_generator.get_state()})
    if method == "bce":
        rank_norm, rank_value, rank_lambda, stats = 0., 0., 0., []
    else:
        draws = int(config["rl_calibration_draws"]) if method in RL_METHODS else 1
        logits = head(x)
        losses, stats = [], []
        for _ in range(draws):
            loss, item = rank_loss(logits, y, eligible, method, tau,
                                   generator=calibration_generator,
                                   group_size=int(config["group_size"]), chunk_size=int(config["chunk_size"]))
            losses.append(loss)
            stats.append(item)
        rank = torch.stack(losses).mean()
        rank_norm = _norm(torch.autograd.grad(rank, parameters))
        rank_value = float(rank.detach())
        require(math.isfinite(rank_norm) and rank_norm > 1e-12,
                "Initial rank gradient is nonfinite or zero; do not silently disable a study arm")
        rank_lambda = float(config["kappa"]) * bce_norm / rank_norm
    train_after = tensor_sha({"state": training_generator.get_state()})
    require(train_before == train_after, "Calibration consumed the training sampling stream")
    report = {"temperature": tau.cpu().tolist(), "temperature_recipe": ".5 * population std(initial train eligible logits), clamp [.25,2]",
              "calibration_source_split": "train", "calibration_training_patients": int(x.shape[0]),
              "calibration_eligible_counts": eligible.sum(dim=0).cpu().tolist(),
              "calibration_positive_counts": (y * eligible).sum(dim=0).to(torch.int64).cpu().tolist(),
              "bce_gradient_norm": bce_norm, "rank_gradient_norm": rank_norm,
              "rank_loss_at_initial": rank_value, "lambda": rank_lambda, "kappa": config["kappa"],
              "rank_gradient_draws": len(stats), "rank_stats": stats,
              "training_generator_seed": int(config["seed"]) + 17041,
              "calibration_generator_seed": int(config["seed"]) + 27041,
              "training_generator_before": train_before, "training_generator_after_calibration": train_after,
              "calibration_generator_before": calibration_before,
              "calibration_generator_after": tensor_sha({"state": calibration_generator.get_state()})}
    return tau, rank_lambda, report


def _export_checkpoint(path, reference, head, config, provenance, epoch, score):
    state = dict(reference["model_state_dict"])
    for name, tensor in _head_state(head).items():
        state[f"classification_head.{name}"] = tensor
    checkpoint = dict(reference)
    checkpoint.update(model_state_dict=state, epoch=epoch, validation_mAP=score,
                      encoder_epoch=reference.get("epoch"), encoder_training_config=reference.get("config"),
                      config=config, ranking_study={"method": config["method"], "head_epoch": epoch,
                                                   "provenance": provenance})
    torch.save(checkpoint, path)


def train_head(config, *, root, provenance, tracker=None):
    """Train one arm, keeping theta0 in the common validation selection set."""
    root = Path(root)
    output = root / config["output_dir"]
    require(not output.exists() or not any(output.iterdir()), f"Run already has evidence: {output}")
    method, seed = config["method"], int(config["seed"])
    require(method in METHODS, "Unknown method")
    device = torch.device(config["device"])
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable; no silent CPU fallback")
    strict_seed(seed, device)
    manifest, reference_info, reference, initial_head, arrays = load_feature_cache(root / config["feature_dir"], root)
    require(manifest["seed"] == seed and reference_info["seed"] == seed, "Feature seed mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", provenance)
    codes = reference_info["target_codes"]
    head = nn.Linear(initial_head["weight"].shape[1], len(codes), device="cpu", dtype=torch.float32)
    head.load_state_dict(initial_head)
    head.to(device)
    initial = {name: value.to(device) for name, value in initial_head.items()}
    data = {split: _tensor_data(row, device) for split, row in arrays.items()}
    train_generator = torch.Generator(device=device).manual_seed(seed + 17041)
    calibration_generator = torch.Generator(device=device).manual_seed(seed + 27041)
    tau, rank_lambda, calibration = calibrate(head, data["train"], method, config,
                                             train_generator, calibration_generator)
    write_json(output / "initialization.json", {"theta0_sha256": tensor_sha(initial_head),
                                              "feature_manifest_sha256": file_sha(root / config["feature_dir"] / "manifest.json"),
                                              "calibration_train_cache_sha256": manifest["files_sha256"]["train.npz"],
                                              **calibration})
    torch.save(initial_head, output / "initial_head.pt")
    write_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "deterministic_enabled": torch.are_deterministic_algorithms_enabled(),
               "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
               "amp_enabled": False, "dtype": "float32", "dropout": False,
               "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "float32_matmul_precision": torch.get_float32_matmul_precision(),
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "optimizer": "Adam", "torch_num_threads": torch.get_num_threads()})
    optimizer = torch.optim.Adam(head.parameters(), lr=float(config["learning_rate"]), weight_decay=0.)
    start = time.perf_counter()
    best_score, best_epoch, stale = -math.inf, 0, 0
    theta0_metrics, fixed10_metrics, last_epoch = None, None, 0
    x, y, eligible = data["train"]["features"], data["train"]["labels"], ~data["train"]["prior"]
    for epoch in range(int(config["epochs"]) + 1):
        diagnostics = {"rank_loss_pre_update": None, "macro_bce_pre_update": None,
                       "ridge_pre_update": None, "objective_pre_update": None, "gradient_norm": None}
        rank_stats = {}
        if epoch:
            head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            bce = macro_bce_logits(logits, y, eligible)
            rank, rank_stats = rank_loss(logits, y, eligible, method, tau, generator=train_generator,
                                         group_size=int(config["group_size"]), chunk_size=int(config["chunk_size"]))
            ridge = ridge_penalty(head, initial)
            loss = bce + rank_lambda * rank + float(config["ridge_mu"]) * ridge
            require(bool(torch.isfinite(loss)), "Nonfinite head objective")
            loss.backward()
            gradients = tuple(p.grad for p in head.parameters())
            norm = _norm(gradients)
            require(math.isfinite(norm), "Nonfinite head gradients")
            diagnostics = {"rank_loss_pre_update": float(rank.detach()), "macro_bce_pre_update": float(bce.detach()),
                           "ridge_pre_update": float(ridge.detach()), "objective_pre_update": float(loss.detach()),
                           "gradient_norm": norm}
            optimizer.step()
        train_metrics, _ = evaluate_head(head, data["train"], codes)
        val_metrics, val_scores = evaluate_head(head, data["val"], codes)
        row = {"epoch": epoch, "lambda": rank_lambda, "learning_rate": float(config["learning_rate"]),
               "elapsed_seconds": time.perf_counter() - start, **diagnostics,
               "sampling_rng_sha256": tensor_sha({"state": train_generator.get_state()})}
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in ("mAP", "macro_auroc", "micro_ap", "brier", "macro_bce", "at_risk_bce"):
                row[f"{split}/{key}"] = metrics[key]
        row.update({f"rank/{key}": value for key, value in rank_stats.items()})
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if tracker is not None:
            tracker.log({key: value for key, value in row.items() if key != "sampling_rng_sha256"}, step=epoch)
        print(json.dumps({"method": method, "seed": seed, "epoch": epoch, "val/mAP": row["val/mAP"],
                          "val/BCE": row["val/at_risk_bce"], "rank_loss": diagnostics["rank_loss_pre_update"]}), flush=True)
        if epoch == 0:
            if "val_initial_mAP" in manifest["checks"]:
                require(abs(val_metrics["mAP"] - manifest["checks"]["val_initial_mAP"]) <= 1e-10,
                        "GPU/CPU initial-head evaluation differs from the frozen cache mAP; stop before any update")
            theta0_metrics = val_metrics
            write_json(output / "theta0_validation_metrics.json", val_metrics)
        if epoch == 10:
            fixed10_metrics = val_metrics
            write_json(output / "epoch10_validation_metrics.json", val_metrics)
            torch.save(_head_state(head), output / "epoch10_head.pt")
            write_predictions(output / "epoch10_val_predictions.csv", arrays["val"]["patient_ids"].astype(str).tolist(),
                              val_scores, codes, expected_patient_ids=arrays["val"]["patient_ids"].astype(str).tolist())
        score = val_metrics["mAP"]
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(_head_state(head), output / "best_head.pt")
            _export_checkpoint(output / "best.pt", reference, head, config, provenance, epoch, score)
            write_json(output / "validation_metrics.json", val_metrics)
        elif epoch:
            stale += 1
        last_epoch = epoch
        if stale >= int(config["patience"]):
            break
    head.load_state_dict(torch.load(output / "best_head.pt", map_location=device, weights_only=True))
    for split in ("train", "val", "test"):
        _, scores = evaluate_head(head, data[split], codes)
        ids = arrays[split]["patient_ids"].astype(str).tolist()
        write_predictions(output / f"{split}_predictions.csv", ids, scores, codes, expected_patient_ids=ids)
    if "history_group_long" in arrays["val"]:
        _, scores = evaluate_head(head, data["val"], codes)
        groups = {}
        for name, mask in (("long", arrays["val"]["history_group_long"].astype(bool)),
                           ("short", ~arrays["val"]["history_group_long"].astype(bool))):
            if mask.any():
                groups[name] = evaluate_predictions(arrays["val"]["labels"][mask], scores[mask], codes)
        write_json(output / "group_validation_metrics.json", groups)
    exported = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    for name, tensor in reference["model_state_dict"].items():
        if not name.startswith("classification_head."):
            require(torch.equal(tensor, exported["model_state_dict"][name]), "Export changed frozen backbone")
    summary = {"method": method, "seed": seed, "best_epoch": best_epoch, "last_epoch": last_epoch,
               "no_update_selected": best_epoch == 0, "best_validation_mAP": best_score,
               "theta0_validation_mAP": theta0_metrics["mAP"],
               "mAP_delta_from_theta0": best_score - theta0_metrics["mAP"],
               "epoch10_validation_mAP": fixed10_metrics["mAP"] if fixed10_metrics else None,
               "lambda": rank_lambda, "calibration": calibration,
               "elapsed_seconds": time.perf_counter() - start, "checkpoint_sha256": file_sha(output / "best.pt"),
               "frozen_backbone_export_verified": True, "test_metrics": None,
               "selection_note": "Epoch0 is eligible for every arm; fallback is not improvement. Primary comparison is against matched-budget BCE head refitting, not only theta0."}
    write_json(output / "summary.json", summary)
    if tracker is not None:
        tracker.summary.update({key: value for key, value in summary.items() if key != "calibration"})
    return summary
