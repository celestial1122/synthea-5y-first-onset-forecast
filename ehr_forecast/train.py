"""Shared training/evaluation loop; no validation or test gradients."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import evaluate_predictions
from .models import build_model
from .submission import write_predictions


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, deterministic_mode: str = "warn_only") -> None:
    if deterministic_mode not in {"warn_only", "strict"}:
        raise ValueError("deterministic_mode must be 'warn_only' or 'strict'")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=deterministic_mode == "warn_only")
    torch.set_num_threads(min(8, os.cpu_count() or 1))


class PatientDataset(Dataset):
    def __init__(self, path: Path, include_row_indices: bool = False):
        with np.load(path, allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        required = {"patient_ids", "tokens", "time_days", "static", "prior_conditions"}
        if not required.issubset(self.arrays):
            raise ValueError(f"Missing prepared arrays: {required - self.arrays.keys()}")
        self.patient_ids = self.arrays["patient_ids"].astype(str).tolist()
        self.include_row_indices = include_row_indices
        if len(set(self.patient_ids)) != len(self.patient_ids) or not self.patient_ids:
            raise ValueError("Patient IDs must be unique and nonempty")
        self.tensors = {
            key: torch.from_numpy(value) for key, value in self.arrays.items()
            if key != "patient_ids" and value.dtype.kind in "biuf"
        }
        if any(len(value) != len(self.patient_ids) for value in self.tensors.values()):
            raise ValueError("Prepared arrays have inconsistent patient counts")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        row = {key: value[index] for key, value in self.tensors.items()}
        if self.include_row_indices:
            row["_row_index"] = torch.tensor(index, dtype=torch.int64)
        return row


def collate_patients(batch):
    result = {key: torch.stack([row[key] for row in batch]) for key in batch[0]}
    if result["tokens"].ndim == 3:
        present = result["tokens"].ne(0)
        length = max(1, int(present.any(dim=-1).sum(dim=1).max()))
        events = max(1, int(present.sum(dim=-1).max()))
        result["tokens"] = result["tokens"][:, :length, :events]
    elif result["tokens"].ndim == 2:
        length = max(1, int(result["tokens"].ne(0).sum(dim=1).max()))
        result["tokens"] = result["tokens"][:, :length]
    else:
        raise ValueError("Prepared tokens must be patient/event or patient/unit/event arrays")
    result["time_days"] = result["time_days"][:, :length]
    return result


def validation_group_masks(path: str | None, patient_ids: list[str]) -> dict:
    if path is None:
        return {}
    assignments = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "val":
                continue
            patient = row["patient_id"]
            if patient in assignments:
                raise ValueError("Duplicate validation patient in history groups")
            assignments[patient] = row["history_group_256"]
    if set(assignments) != set(patient_ids):
        raise ValueError("History groups must cover exactly the official validation patients")
    return {name: np.array([assignments[patient] == name for patient in patient_ids])
            for name in sorted(set(assignments.values()))}


def make_loader(dataset, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      collate_fn=collate_patients, generator=generator)


def choose_device(requested: str) -> torch.device:
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def model_logits(model, batch):
    return model(batch["tokens"].long(), batch["time_days"].float(), batch["static"].float())


@torch.inference_mode()
def predict_probabilities(model, loader, device):
    model.eval()
    predictions = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model_logits(model, batch)
        offset = getattr(model, "prediction_logit_offset", None)
        if offset is not None:
            # Weighted BCE estimates a tilted posterior. Undo its known log-odds
            # shift before presenting probabilities; finite-model calibration
            # is still assessed on validation and is not guaranteed by this.
            logits = logits - offset
        probabilities = torch.sigmoid(logits.float())
        # An already diagnosed condition cannot be a first-ever future diagnosis.
        probabilities.masked_fill_(batch["prior_conditions"].bool(), 0.0)
        predictions.append(probabilities.cpu().numpy())
    return np.concatenate(predictions)


@torch.inference_mode()
def language_model_loss(model, loader, device):
    model.eval()
    total, weight = 0.0, 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        count = int(batch["tokens"][:, 1:].ne(0).sum())
        loss = model.next_event_loss(batch["tokens"].long(), batch["time_days"].float(), batch["static"].float())
        total += float(loss) * count
        weight += count
    return total / max(weight, 1)


def _curve_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def _training_step(loss, optimizer, scaler, parameters, collect_stats=False):
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite loss: {loss.item()}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    scale_before = scaler.get_scale() if collect_stats else None
    scaler.step(optimizer)
    scaler.update()
    if collect_stats:
        return {"pre_clip_norm": float(norm), "clipped": bool(norm > 1),
                "amp_step_skipped": bool(scaler.is_enabled() and scaler.get_scale() < scale_before)}


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_fingerprints():
    states = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        states.update({f"cuda_{index}": state for index, state in enumerate(torch.cuda.get_rng_state_all())})
    return {name: state_fingerprint({name: state}) for name, state in states.items()}


def at_risk_bce(labels, probabilities, prior_conditions):
    scores = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    truth = np.asarray(labels)
    eligible = ~np.asarray(prior_conditions, dtype=bool)
    if not eligible.any():
        return None
    return float((-(truth * np.log(scores) + (1 - truth) * np.log1p(-scores)))[eligible].mean())


def train(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    deterministic_mode = config.get("deterministic_mode", "warn_only")
    seed_everything(seed, deterministic_mode)
    prepared = Path(config["prepared_dir"])
    output = Path(config["output_dir"])
    if (output / "best.pt").exists() or (output / "history.jsonl").exists():
        raise FileExistsError(f"Run directory already contains a run; choose a new output_dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    diagnostics = bool(config.get("audit_diagnostics", False))
    data = {split: PatientDataset(prepared / f"{split}.npz", include_row_indices=diagnostics) for split in ("train", "val", "test")}
    group_masks = validation_group_masks(config.get("validation_groups_csv"), data["val"].patient_ids)
    if any("labels" not in data[split].arrays for split in ("train", "val")):
        raise ValueError("Train and validation labels are required")
    if "labels" in data["test"].arrays:
        raise ValueError("Test labels must not be present")
    id_sets = [set(dataset.patient_ids) for dataset in data.values()]
    if any(id_sets[a] & id_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Patient overlap between official splits")
    for split in ("train", "val"):
        labels, prior = data[split].arrays["labels"], data[split].arrays["prior_conditions"]
        if np.any(labels[prior.astype(bool)]):
            raise ValueError(f"Incident labels overlap prior diagnoses in {split}")
    device = choose_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_size = int(config.get("batch_size", 32))
    train_loader = make_loader(data["train"], batch_size, shuffle=True, seed=seed)
    eval_loaders = {split: make_loader(dataset, batch_size * 2) for split, dataset in data.items()}
    model_config = {key: config[key] for key in ("model_type", "d_model", "n_heads", "n_layers", "dropout", "max_unit_events", "window_size", "window_stride", "local_layers", "global_layers", "bag_normalization", "attention_mode", "readout_mode") if key in config}
    model_config.update(vocab_size=metadata["vocab_size"], n_targets=len(metadata["target_codes"]),
                        n_static=metadata["n_static"], max_events=int(config.get("model_max_events", metadata["max_events"])))
    if model_config["max_events"] < metadata["max_events"]:
        raise ValueError("Model position capacity cannot be smaller than prepared sequence length")
    positives = data["train"].arrays["labels"].sum(axis=0)
    at_risk = (~data["train"].arrays["prior_conditions"].astype(bool)).sum(axis=0)
    priors = (positives + 0.5) / (at_risk + 1.0)
    weight_policy = config.get("loss_positive_weight", "none")
    if weight_policy not in {"none", "sqrt"}:
        raise ValueError("loss_positive_weight must be 'none' or 'sqrt'")
    positive_weights = np.ones_like(priors)
    if weight_policy == "sqrt":
        positive_weights = np.clip(np.sqrt((at_risk - positives + 1) / (positives + 1)), 1, 8)
        model_config["prediction_logit_offset"] = np.log(positive_weights).tolist()
    model = build_model(model_config).to(device)
    # Classifier starts at train-only incidence priors, rather than 50% for rare diseases.
    head = getattr(model, "classification_head", None)
    if isinstance(head, nn.Linear) and head.bias is not None:
        with torch.no_grad():
            head.bias.copy_(torch.tensor(np.log(priors / (1 - priors)) + np.log(positive_weights), device=device, dtype=head.bias.dtype))
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed, "deterministic_algorithms": f"enabled_{deterministic_mode}",
        "deterministic_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_num_threads": torch.get_num_threads(),
        "amp_enabled": bool(config.get("amp", True)) and device.type == "cuda",
        "amp_dtype": ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.type == "cuda" else None,
        "sdp_backends_enabled": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                                 "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 "math": torch.backends.cuda.math_sdp_enabled()},
    }
    provenance = {path.name: sha256_file(path) for path in prepared.iterdir() if path.is_file()}
    write_json(output / "config.json", config)
    write_json(output / "model_config.json", model_config)
    write_json(output / "environment.json", environment)
    write_json(output / "prepared_sha256.json", provenance)
    if diagnostics:
        write_json(output / "initial_state.json", {"state_sha256": state_fingerprint(model.state_dict()),
                                                   "rng": rng_fingerprints(),
                                                   "input_shape": list(data["train"].arrays["tokens"].shape),
                                                   "observed_train_token_ids": int(np.unique(data["train"].arrays["tokens"]).size),
                                                   "total_parameters": sum(p.numel() for p in model.parameters())})
        torch.save({"model_state_dict": model.state_dict(), "model_config": model_config, "seed": seed}, output / "initial.pt")
    for name in ("metadata.json", "vocabulary.json", "preprocessing.json", "audit.json"):
        if (prepared / name).exists():
            shutil.copy2(prepared / name, output / name)

    import wandb
    wandb_run = wandb.init(project=config.get("wandb_project", "patient-timeline-forecast"),
                           name=config.get("name", output.name), mode=config.get("wandb_mode", "offline"),
                           dir=str(output.resolve()), config={**config, **environment},
                           settings=wandb.Settings(init_timeout=90))
    write_json(output / "tracking.json", {"id": wandb_run.id, "mode": config.get("wandb_mode", "offline"),
                                         "url": wandb_run.url, "dir": wandb_run.dir})
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    pretrain_epochs = int(config.get("pretrain_epochs", 0))
    if pretrain_epochs and config.get("model_type") not in {"transformer", "pooled_transformer"}:
        raise ValueError("Next-event pretraining is supported only for the Transformer")
    start = time.perf_counter()
    step = 0
    if diagnostics:
        write_json(output / "training_begin_rng.json", rng_fingerprints())
    try:
        if pretrain_epochs:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("pretrain_learning_rate", 5e-4),
                                          weight_decay=config.get("weight_decay", 0.01))
            for epoch in range(1, pretrain_epochs + 1):
                model.train()
                total, count = 0.0, 0
                for raw_batch in train_loader:
                    batch = move_batch(raw_batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                        loss = model.next_event_loss(batch["tokens"].long(), batch["time_days"].float(), batch["static"].float())
                    _training_step(loss, optimizer, scaler, model.parameters())
                    weight = int(batch["tokens"][:, 1:].ne(0).sum())
                    total += float(loss.detach()) * weight
                    count += weight
                val_loss = language_model_loss(model, eval_loaders["val"], device)
                row = {"stage": "pretrain", "epoch": epoch, "pretrain/train_loss": total / max(count, 1),
                       "pretrain/val_loss": val_loss, "elapsed_seconds": time.perf_counter() - start}
                step += 1
                _curve_row(output / "history.jsonl", row)
                wandb_run.log({key: value for key, value in row.items() if key != "stage"}, step=step)
                print(json.dumps(row), flush=True)
            torch.save({"model_state_dict": model.state_dict(), "model_config": model_config}, output / "pretrained.pt")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4),
                                      weight_decay=config.get("weight_decay", 0.01))
        epochs = int(config.get("epochs", 35))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=(torch.as_tensor(positive_weights, dtype=torch.float32, device=device) if weight_policy == "sqrt" else None))
        best_score, best_epoch, stale = -math.inf, 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, loss_count = 0.0, 0
            step_diagnostics = []
            order_digest = hashlib.sha256()
            for raw_batch in train_loader:
                if diagnostics:
                    order_digest.update(raw_batch["_row_index"].numpy().astype("<i8").tobytes())
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    logits = model_logits(model, batch)
                    eligible = ~batch["prior_conditions"].bool()
                    losses = criterion(logits, batch["labels"].float())
                    loss = (losses * eligible).sum() / eligible.sum().clamp_min(1)
                step_result = _training_step(loss, optimizer, scaler, model.parameters(), collect_stats=diagnostics)
                if diagnostics:
                    step_diagnostics.append(step_result)
                    if epoch == 1 and len(step_diagnostics) == 1:
                        write_json(output / "supervised_parameter_audit.json", {
                            "with_gradient": {name: p.numel() for name, p in model.named_parameters() if p.grad is not None},
                            "without_gradient": {name: p.numel() for name, p in model.named_parameters() if p.grad is None},
                            "note": "Gradient-bearing tensor size is not effective capacity; unused position/token rows remain inside these tensors.",
                        })
                weight = int(eligible.sum())
                loss_sum += float(loss.detach()) * weight
                loss_count += weight
            scheduler.step()
            val_probabilities = predict_probabilities(model, eval_loaders["val"], device)
            metrics = evaluate_predictions(data["val"].arrays["labels"], val_probabilities,
                                           metadata["target_codes"], metadata.get("target_descriptions"))
            score = metrics["mAP"]
            eligible_val = ~data["val"].arrays["prior_conditions"].astype(bool)
            clipped = np.clip(val_probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
            truth = data["val"].arrays["labels"]
            val_bce = float((-(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)))[eligible_val].mean())
            row = {"stage": "finetune", "epoch": epoch, "train/loss": loss_sum / max(loss_count, 1),
                   "val/mAP": score, "val/macro_auroc": metrics["macro_auroc"],
                   "val/loss": val_bce,
                   "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - start}
            if diagnostics:
                train_probabilities = predict_probabilities(model, eval_loaders["train"], device)
                train_metrics = evaluate_predictions(data["train"].arrays["labels"], train_probabilities,
                                                     metadata["target_codes"], metadata.get("target_descriptions"))
                row.update({"train_eval/BCE": at_risk_bce(data["train"].arrays["labels"], train_probabilities,
                                                         data["train"].arrays["prior_conditions"]),
                            "train_eval/mAP": train_metrics["mAP"], "train_eval/macro_auroc": train_metrics["macro_auroc"],
                            "train_eval/Brier": train_metrics["brier"],
                            "gradient/norm_mean": float(np.mean([item["pre_clip_norm"] for item in step_diagnostics])),
                            "gradient/norm_max": max(item["pre_clip_norm"] for item in step_diagnostics),
                            "gradient/clip_fraction": float(np.mean([item["clipped"] for item in step_diagnostics])),
                            "amp/skipped_steps": sum(item["amp_step_skipped"] for item in step_diagnostics),
                            "train_patient_order_sha256": order_digest.hexdigest()})
            group_results = {}
            for name, mask in group_masks.items():
                group_metrics = evaluate_predictions(truth[mask], val_probabilities[mask],
                                                       metadata["target_codes"], metadata.get("target_descriptions"))
                group_results[name] = group_metrics
                row[f"val_{name}/mAP"] = group_metrics["mAP"]
                row[f"val_{name}/macro_auroc"] = group_metrics["macro_auroc"]
            _curve_row(output / "history.jsonl", row)
            step += 1
            wandb_run.log({key: value for key, value in row.items() if key not in {"stage", "train_patient_order_sha256"}}, step=step)
            print(json.dumps(row), flush=True)
            if score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                checkpoint = {"model_state_dict": model.state_dict(), "model_config": model_config,
                              "target_codes": metadata["target_codes"], "epoch": epoch,
                              "validation_mAP": score, "seed": seed, "config": config,
                              "prepared_sha256": provenance}
                torch.save(checkpoint, output / "best.pt")
                write_json(output / "validation_metrics.json", metrics)
                if group_results:
                    write_json(output / "group_validation_metrics.json", group_results)
            else:
                stale += 1
            if stale >= int(config.get("patience", 8)):
                break

        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        for split in ("val", "test"):
            probabilities = predict_probabilities(model, eval_loaders[split], device)
            write_predictions(output / f"{split}_predictions.csv", data[split].patient_ids, probabilities,
                              metadata["target_codes"], expected_patient_ids=data[split].patient_ids)
        val_baseline = np.tile(priors, (len(data["val"]), 1))
        val_baseline[data["val"].arrays["prior_conditions"].astype(bool)] = 0
        write_json(output / "prevalence_baseline_metrics.json", evaluate_predictions(
            data["val"].arrays["labels"], val_baseline, metadata["target_codes"], metadata.get("target_descriptions")))
        summary = {"best_epoch": best_epoch, "best_validation_mAP": best_score,
                   "elapsed_seconds": time.perf_counter() - start, "test_patients": len(data["test"]),
                   "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                   "test_labels_available": False, "test_metrics": None,
                   "checkpoint_sha256": sha256_file(output / "best.pt")}
        write_json(output / "summary.json", summary)
        wandb_run.summary.update(summary)
        _write_per_condition_csv(output / "validation_metrics.json", output / "per_condition.csv")
        plot_curves(output)
        return summary
    finally:
        wandb_run.finish()


def _write_per_condition_csv(metrics_path, destination):
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))["per_condition"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    supervised = [row for row in rows if row["stage"] == "finetune"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, key, title in zip(axes, ("train/loss", "val/mAP", "val/macro_auroc"),
                                ("Training at-risk BCE", "Validation mAP (all targets)", "Validation macro AUROC (valid targets)")):
        axis.plot([row["epoch"] for row in supervised], [row[key] for row in supervised])
        axis.set(xlabel="Fine-tuning epoch", ylabel="BCE loss" if key == "train/loss" else "Score (0-1)", title=title)
        axis.grid(alpha=0.2)
    fig.suptitle(f"{output.name} | official patient split; source: history.jsonl", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)


def predict(checkpoint_path: Path, prepared_dir: Path, output_path: Path, device="auto", split="test") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    if checkpoint["target_codes"] != metadata["target_codes"]:
        raise ValueError("Checkpoint target order does not match prepared data")
    for name in ("metadata.json", "vocabulary.json", "preprocessing.json"):
        expected = checkpoint.get("prepared_sha256", {}).get(name)
        if expected and sha256_file(prepared_dir / name) != expected:
            raise ValueError(f"Checkpoint preprocessing does not match {name}")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_device = choose_device(device)
    model.to(selected_device)
    dataset = PatientDataset(prepared_dir / f"{split}.npz")
    probabilities = predict_probabilities(model, make_loader(dataset, 64), selected_device)
    write_predictions(output_path, dataset.patient_ids, probabilities, metadata["target_codes"],
                      expected_patient_ids=dataset.patient_ids)
    return {"patients": len(dataset), "conditions": len(metadata["target_codes"]), "path": str(output_path)}
