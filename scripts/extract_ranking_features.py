"""Cache frozen causal last-state features for a head-only ranking study.

The extractor never trains or changes fitted evidence. A cache is only reused
when its complete provenance and every saved file agree; partial caches fail.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ehr_forecast.metrics import evaluate_predictions
from ehr_forecast.models import build_model
from ehr_forecast.train import (
    PatientDataset, choose_device, make_loader, model_logits, move_batch,
    seed_everything, sha256_file,
)

AUDIT = ROOT / "artifacts/experimental_audit_strict"
PREPARED = ROOT / "artifacts/optimization/prepared_l512"
OUTPUT = ROOT / "artifacts/group_ranking/features"
SEEDS = (17, 42, 73)
SPLITS = ("train", "val", "test")
BATCH_SIZE = 32
FEATURE_DIM = 133
PROBABILITY_TOLERANCE = 1e-6


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_saved_predictions(path, patient_ids, codes):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    require(rows[0] == ["patient_id", *codes], f"Prediction target order mismatch: {path}")
    require([row[0] for row in rows[1:]] == list(patient_ids), f"Prediction patient order mismatch: {path}")
    p = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
    require(p.shape == (len(patient_ids), len(codes)) and np.isfinite(p).all()
            and (p >= 0).all() and (p <= 1).all(), f"Invalid probabilities: {path}")
    return p


def verify_frozen_inputs():
    manifest = read(AUDIT / "frozen_manifest.json")
    require(manifest["deterministic_mode"] == "strict" and manifest["seeds"] == list(SEEDS), "Wrong strict audit")
    for name, expected in manifest["source_sha256"].items():
        require(sha256_file(ROOT / name) == expected, f"Frozen source changed: {name}")
    for name, expected in manifest["prepared_sha256"].items():
        require(sha256_file(PREPARED / name) == expected, f"Prepared data changed: {name}")
    for name, expected in manifest["auxiliary_inputs_sha256"].items():
        require(sha256_file(ROOT / name) == expected, f"Frozen auxiliary input changed: {name}")
    for path_key, hash_key in (("original_protocol_path", "original_protocol_sha256"),
                               ("strict_protocol_path", "strict_protocol_sha256"),
                               ("original_frozen_manifest_path", "original_frozen_manifest_sha256")):
        require(sha256_file(ROOT / manifest[path_key]) == manifest[hash_key], f"Frozen protocol/manifest changed: {path_key}")
    summary = read(AUDIT / "audit_summary.json")
    require(summary["strict"] and summary["factorial_controls"]["passed"]
            and not summary["uses_probability_ensemble"], "Strict audit controls not passed")
    return manifest, summary


def load_prepared():
    metadata = read(PREPARED / "metadata.json")
    require(metadata["n_static"] == 5 and len(metadata["target_codes"]) == 40, "Expected five static features and 40 targets")
    data = {split: PatientDataset(PREPARED / f"{split}.npz") for split in SPLITS}
    for split, dataset in data.items():
        require(("labels" in dataset.arrays) == (split != "test"), f"Unexpected labels for {split}")
        require(dataset.arrays["static"].shape[1] == 5, "Static feature shape mismatch")
        if split != "test":
            require(not np.any(dataset.arrays["labels"][dataset.arrays["prior_conditions"].astype(bool)]), "Incident/prior overlap")
    require(all(not (set(data[a].patient_ids) & set(data[b].patient_ids))
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))), "Patient split overlap")
    val_long = int((data["val"].arrays["full_history_length"] > 255).sum())
    require(val_long == 133 and len(data["val"]) - val_long == 232, "Frozen validation groups changed")
    return metadata, data


def runtime(device):
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                device=str(device), cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                inference_dtype="float32", autocast=False, deterministic_mode="strict", batch_size=BATCH_SIZE)


def reference_for(seed, manifest, summary, metadata, device):
    key = f"C0R0-seed{seed}"
    run = summary["runs"][key]
    folder = ROOT / run["folder"]
    require(folder.resolve() == (AUDIT / f"runs/C0R0-seed{seed}-repeat-a").resolve(), "Baseline must be fixed repeat-a")
    for filename, expected in run["source_sha256"].items():
        require(sha256_file(folder / filename) == expected, f"Audited reference changed: {folder / filename}")
    checkpoint_path = folder / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["target_codes"] == metadata["target_codes"], "Reference target order mismatch")
    config = checkpoint["model_config"]
    require(config["model_type"] == "transformer" and config["d_model"] == 128
            and config["n_static"] == 5 and config["n_targets"] == 40 and config["max_events"] == 1024,
            "Reference is not the frozen strong causal model")
    require(read(folder / "prepared_sha256.json") == manifest["prepared_sha256"], "Reference prepared inputs changed")
    provenance = dict(reference_checkpoint_path=relative(checkpoint_path), reference_checkpoint_sha256=sha256_file(checkpoint_path),
                      model_config=config, target_codes=metadata["target_codes"],
                      prepared_dir=relative(PREPARED), prepared_sha256=manifest["prepared_sha256"],
                      strict_frozen_manifest_sha256=sha256_file(AUDIT / "frozen_manifest.json"),
                      strict_source_sha256=manifest["source_sha256"],
                      audit_summary_sha256=sha256_file(AUDIT / "audit_summary.json"),
                      extractor_script_path=relative(Path(__file__)), extractor_script_sha256=sha256_file(Path(__file__)),
                      feature_recipe="concat(model.encode(tokens,time)[last_non_PAD_position], static); eval mode; no dropout/no AMP",
                      feature_dim=FEATURE_DIM, runtime=runtime(device), seed=seed)
    reference = dict(path=relative(checkpoint_path), sha256=sha256_file(checkpoint_path), seed=seed,
                     reference_run_dir=relative(folder), model_config=config, target_codes=metadata["target_codes"],
                     validation_mAP=checkpoint["validation_mAP"], epoch=checkpoint["epoch"])
    return folder, checkpoint, provenance, reference


@torch.inference_mode()
def feature_batch(model, batch):
    """Same pre-classifier vector as TimelineTransformer.forward in eval mode."""
    tokens, days, static = batch["tokens"].long(), batch["time_days"].float(), batch["static"].float()
    hidden = model.encode(tokens, days)
    positions = tokens.ne(0).sum(dim=1).sub(1).clamp_min(0)
    last = hidden[torch.arange(len(tokens), device=tokens.device), positions]
    return torch.cat((last, static.to(last.dtype)), dim=-1)


@torch.inference_mode()
def initial_probabilities(features, head, prior):
    logits = F.linear(torch.as_tensor(features, dtype=torch.float32), head["weight"], head["bias"])
    p = torch.sigmoid(logits.float()).numpy()
    p[np.asarray(prior, dtype=bool)] = 0
    return p


@torch.inference_mode()
def extract_split(model, dataset, device, head):
    require(not model.training, "Feature extraction requires eval mode")
    features, full_probabilities = [], []
    logit_error, probability_error = 0.0, 0.0
    for raw in make_loader(dataset, BATCH_SIZE, shuffle=False):
        batch = move_batch(raw, device)
        with torch.autocast(device_type=device.type, enabled=False):
            feature = feature_batch(model, batch)
            logits_from_features = F.linear(feature, model.classification_head.weight, model.classification_head.bias)
            full_logits = model_logits(model, batch)
        require(feature.dtype == torch.float32 and feature.shape[1] == FEATURE_DIM, "Wrong feature dtype/dimension")
        logit_error = max(logit_error, float((logits_from_features - full_logits).abs().max()))
        full_p, linear_p = torch.sigmoid(full_logits.float()), torch.sigmoid(logits_from_features.float())
        probability_error = max(probability_error, float((full_p - linear_p).abs().max()))
        full_p.masked_fill_(batch["prior_conditions"].bool(), 0)
        features.append(feature.cpu().numpy())
        full_probabilities.append(full_p.cpu().numpy())
    feature = np.concatenate(features).astype(np.float32, copy=False)
    full_p = np.concatenate(full_probabilities)
    require(feature.shape == (len(dataset), FEATURE_DIM) and np.isfinite(feature).all(), "Invalid cached features")
    cpu_p = initial_probabilities(feature, head, dataset.arrays["prior_conditions"])
    cpu_error = float(np.max(np.abs(cpu_p - full_p)))
    require(logit_error <= 1e-6 and probability_error <= PROBABILITY_TOLERANCE and cpu_error <= PROBABILITY_TOLERANCE,
            f"Full forward vs feature/Linear equivalence failed: logit={logit_error}, CPU p={cpu_error}")
    fields = {name: dataset.arrays[name].copy() for name in
              ("patient_ids", "prior_conditions", "full_history_length", "anchor_dates")}
    fields.update(features=feature, history_group_long=dataset.arrays["full_history_length"] > 255)
    if "labels" in dataset.arrays:
        fields["labels"] = dataset.arrays["labels"].astype(np.float32, copy=True)
    checks = dict(full_forward_vs_device_linear_max_logit_error=logit_error,
                  full_forward_vs_device_linear_max_probability_error=probability_error,
                  full_forward_vs_CPU_cached_linear_max_probability_error=cpu_error,
                  patient_order_preserved=True, features_finite_float32=True)
    return fields, cpu_p, full_p, checks


def validate_cache(directory, provenance, reference, datasets, metadata, checkpoint):
    manifest = read(directory / "manifest.json")
    require(manifest["schema_version"] == 1 and manifest["seed"] == provenance["seed"]
            and manifest["provenance"] == provenance, f"Feature cache provenance mismatch: {directory}")
    require(set(manifest["files_sha256"]) == {"train.npz", "val.npz", "test.npz", "initial_head.pt", "reference_checkpoint.json"}, "Incomplete cache file hash coverage")
    for name, expected in manifest["files_sha256"].items():
        require((directory / name).is_file() and sha256_file(directory / name) == expected, f"Feature cache file changed/missing: {name}")
    require(read(directory / "reference_checkpoint.json") == reference, "Cached reference metadata differs")
    head = torch.load(directory / "initial_head.pt", map_location="cpu", weights_only=True)
    require(set(head) == {"weight", "bias"}, "Unexpected initial head fields")
    for name in ("weight", "bias"):
        require(head[name].device.type == "cpu" and head[name].dtype == torch.float32
                and torch.equal(head[name], checkpoint["model_state_dict"][f"classification_head.{name}"]), "Cached head differs from reference")
    for split, dataset in datasets.items():
        with np.load(directory / f"{split}.npz", allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}
        expected = {"features", "patient_ids", "prior_conditions", "full_history_length", "anchor_dates", "history_group_long"}
        if split != "test":
            expected.add("labels")
        require(set(arrays) == expected, f"Wrong feature cache array fields: {split}")
        require(arrays["features"].dtype == np.float32 and arrays["features"].shape == (len(dataset), FEATURE_DIM)
                and np.isfinite(arrays["features"]).all(), f"Invalid cached feature array: {split}")
        for name in expected - {"features", "history_group_long"}:
            require(np.array_equal(arrays[name], dataset.arrays[name]), f"Cached patient data differs: {split}/{name}")
        require(np.array_equal(arrays["history_group_long"], dataset.arrays["full_history_length"] > 255), "Cached history groups differ")
        require(manifest["splits"][split]["patient_ids"] == dataset.patient_ids
                and manifest["splits"][split]["patient_ids_sha256"] == ids_sha(dataset.patient_ids), "Cached ordered IDs differ")
        if split == "val":
            p = initial_probabilities(arrays["features"], head, arrays["prior_conditions"])
            saved = load_saved_predictions(ROOT / reference["reference_run_dir"] / "val_predictions.csv", dataset.patient_ids, metadata["target_codes"])
            maximum = float(np.max(np.abs(p - saved)))
            score = evaluate_predictions(arrays["labels"], p, metadata["target_codes"])["mAP"]
            require(maximum <= PROBABILITY_TOLERANCE and abs(score - reference["validation_mAP"]) <= 1e-12,
                    "Reused cache no longer reconstructs saved validation predictions/mAP")
    require(manifest["checks"]["passed"] is True and manifest["checks"]["val_mAP_equal"] is True
            and manifest["checks"]["val_initial_probs_max_error"] <= PROBABILITY_TOLERANCE, "Saved cache checks not passed")
    return manifest


def extract_seed(seed, requested_device="auto"):
    frozen, summary = verify_frozen_inputs()
    metadata, datasets = load_prepared()
    seed_everything(seed, "strict")
    device = choose_device(requested_device)
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    folder, checkpoint, provenance, reference = reference_for(seed, frozen, summary, metadata, device)
    destination = OUTPUT / f"seed{seed}"
    if destination.exists() and any(destination.iterdir()):
        require((destination / "manifest.json").is_file(), f"Partial cache exists; refusing overwrite: {destination}")
        manifest = validate_cache(destination, provenance, reference, datasets, metadata, checkpoint)
        return dict(seed=seed, reused=True, directory=relative(destination), checks=manifest["checks"])
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device, dtype=torch.float32).eval()
    require(not hasattr(model, "prediction_logit_offset"), "Feature study assumes unweighted original classifier logits")
    for p in model.parameters():
        p.requires_grad_(False)
    head = {name: checkpoint["model_state_dict"][f"classification_head.{name}"].detach().cpu().float().clone()
            for name in ("weight", "bias")}
    require(head["weight"].shape == (40, FEATURE_DIM) and head["bias"].shape == (40,), "Unexpected initial head shape")
    arrays, split_checks, val_cpu, val_full = {}, {}, None, None
    for split in SPLITS:
        fields, cpu_p, full_p, checks = extract_split(model, datasets[split], device, head)
        arrays[split], split_checks[split] = fields, checks
        if split == "val":
            val_cpu, val_full = cpu_p, full_p
    saved = load_saved_predictions(folder / "val_predictions.csv", datasets["val"].patient_ids, metadata["target_codes"])
    maximum = float(np.max(np.abs(val_cpu - saved)))
    full_maximum = float(np.max(np.abs(val_full - saved)))
    val_score = evaluate_predictions(datasets["val"].arrays["labels"], val_cpu, metadata["target_codes"])["mAP"]
    require(maximum <= PROBABILITY_TOLERANCE and full_maximum <= PROBABILITY_TOLERANCE,
            f"Initial feature-head validation predictions differ from saved checkpoint: CPU={maximum}, full={full_maximum}")
    require(abs(val_score - reference["validation_mAP"]) <= 1e-12, "Initial feature-head mAP differs from saved validation mAP")
    # Check the evidence again before any cache file is created.
    again, again_summary = verify_frozen_inputs()
    require(again == frozen and again_summary == summary, "Source/data/audit changed during extraction")
    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        np.savez_compressed(destination / f"{split}.npz", **arrays[split])
    torch.save(head, destination / "initial_head.pt")
    write_json(destination / "reference_checkpoint.json", reference)
    files = ("train.npz", "val.npz", "test.npz", "initial_head.pt", "reference_checkpoint.json")
    manifest = dict(schema_version=1, seed=seed, kind="frozen_causal_last_state_features", provenance=provenance,
                    feature_names=[f"last_hidden_{i:03d}" for i in range(128)] + metadata["static_feature_names"],
                    target_codes=metadata["target_codes"],
                    splits={split: dict(path=f"{split}.npz", n_patients=len(datasets[split]),
                                        feature_shape=list(arrays[split]["features"].shape), dtype="float32",
                                        patient_ids=datasets[split].patient_ids, patient_ids_sha256=ids_sha(datasets[split].patient_ids),
                                        labels_available=split != "test", long_patient_count=int(arrays[split]["history_group_long"].sum()))
                            for split in SPLITS},
                    files_sha256={name: sha256_file(destination / name) for name in files},
                    checks=dict(passed=True, frozen_source_and_data_verified=True,
                                val_initial_probs_max_error=maximum, val_full_forward_probs_max_error=full_maximum,
                                val_initial_mAP=val_score, val_reference_mAP=reference["validation_mAP"],
                                val_mAP_difference=val_score - reference["validation_mAP"], val_mAP_equal=True,
                                full_forward_and_feature_linear_equivalent=True, by_split=split_checks))
    write_json(destination / "manifest.json", manifest)
    validate_cache(destination, provenance, reference, datasets, metadata, checkpoint)
    return dict(seed=seed, reused=False, directory=relative(destination), checks=manifest["checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    require(len(set(args.seeds)) == len(args.seeds) and set(args.seeds).issubset(SEEDS), "Use distinct declared seeds 17/42/73")
    result = [extract_seed(seed, args.device) for seed in args.seeds]
    print(json.dumps(dict(passed=True, features=FEATURE_DIM, runs=result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
