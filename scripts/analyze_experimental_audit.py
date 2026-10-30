"""Gate-controlled, single-model analysis of the frozen 2 x 2 experiment.

Run only after every fit is complete. No model is fitted and no GPU is used.
The script validates controls before computing or writing factorial effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ehr_forecast.factorial_audit import CELLS, factorial_analysis
from ehr_forecast.metrics import evaluate_predictions, validate_probability_matrix


SEEDS = (17, 42, 73)
TREATMENTS = {
    "C0R0": ("causal", "legacy"), "C1R0": ("bidirectional", "legacy"),
    "C0R1": ("causal", "pooled_mlp"), "C1R1": ("bidirectional", "pooled_mlp"),
}
# These fields describe output, observation, or the two declared treatments.
# Every other setting must match, including optimization, AMP and sequence size.
NON_TRAINING_FIELDS = {
    "name", "output_dir", "model_type", "attention_mode", "readout_mode",
    "audit_diagnostics", "validation_groups_csv",
}
SOURCE_NOTE = "Official validation, prepared_l512; no test labels; no probability ensemble."
UNCERTAINTY_NOTE = (
    "Patient bootstrap intervals condition on the three fitted seeds and selected checkpoints. "
    "They do not include refitting/seed uncertainty or undo validation selection. Multiple "
    "contrasts/metrics/groups have no simultaneous correction. Three-seed SD is descriptive, not a seed-population CI. "
    "Long/short intervals are computed separately, without a shared history-group x treatment contrast; "
    "significance in one group but not the other does not establish a history-specific treatment effect."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state):
    """Match train.state_fingerprint without importing training/CUDA helpers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def history(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(all(row.get("stage") == "finetune" for row in rows), f"Unexpected non-finetune history: {path}")
    epochs = [row["epoch"] for row in rows]
    require(epochs == list(range(1, len(rows) + 1)) and rows,
            f"Non-contiguous/duplicate/missing finetune epochs: {path}")
    return rows


def checked_gate(path, seeds=None, row_key=None):
    gate = read_json(path)
    require(gate.get("passed") is True, f"Gate failed: {path}")
    if seeds is not None and row_key:
        rows = gate.get(row_key, [])
        require(len(rows) == len(seeds) and {row["seed"] for row in rows} == set(seeds),
                f"Incomplete/duplicate seed coverage: {path}")
    return gate


def recheck_strict_pair(root, audit, prepared, manifest, gate):
    """Read-only recomputation of the frozen runner's exact A/A checks."""
    first, second = [resolve_path(root, gate[key]) for key in ("baseline_run_dir", "repeat_run_dir")]
    seed = gate["seed"]
    require(first == audit / f"runs/C0R0-seed{seed}-repeat-a"
            and second == audit / f"runs/C0R0-seed{seed}-repeat-b", "Strict baseline must be the predeclared repeat-a")
    required = {
        "config.json", "model_config.json", "environment.json", "prepared_sha256.json", "initial_state.json",
        "initial.pt", "training_begin_rng.json", "history.jsonl", "best.pt", "summary.json", "val_predictions.csv",
        "test_predictions.csv", "validation_metrics.json", "supervised_parameter_audit.json", "strict_provenance.json",
    }
    expected_files = {(path / name).relative_to(root).as_posix() for path in (first, second) for name in required}
    require(set(gate.get("artifact_sha256", {})) == expected_files, "Strict A/A artifact hash coverage is incomplete")
    require(gate.get("frozen_manifest_sha256") == file_sha(audit / "frozen_manifest.json"), "Strict A/A freeze changed")
    for name, expected in gate["artifact_sha256"].items():
        require(file_sha(root / name) == expected, f"Strict A/A artifact changed after verification: {name}")
    configs = [{key: value for key, value in read_json(path / "config.json").items()
                if key not in {"name", "output_dir"}} for path in (first, second)]
    metadata = [read_json(path / "initial_state.json") for path in (first, second)]
    initial_states = [torch.load(path / "initial.pt", map_location="cpu", weights_only=True) for path in (first, second)]
    best = [torch.load(path / "best.pt", map_location="cpu", weights_only=True) for path in (first, second)]
    initial_hash = [state_sha(value["model_state_dict"]) for value in initial_states]
    best_hash = [state_sha(value["model_state_dict"]) for value in best]
    summaries = [read_json(path / "summary.json") for path in (first, second)]
    rows = [[{key: value for key, value in row.items() if key != "elapsed_seconds"}
             for row in history(path / "history.jsonl")] for path in (first, second)]
    environments = [read_json(path / "environment.json") for path in (first, second)]
    predictions_equal = {}
    codes = read_json(prepared / "metadata.json")["target_codes"]
    for split in ("val", "test"):
        with np.load(prepared / f"{split}.npz", allow_pickle=False) as arrays:
            ids = arrays["patient_ids"].astype(str).tolist()
        probabilities = [read_predictions(path / f"{split}_predictions.csv", ids, codes) for path in (first, second)]
        predictions_equal[split] = bool(np.array_equal(*probabilities))
    checks = {
        "configuration_equal_except_name_output": configs[0] == configs[1],
        "initial_state_equal": initial_hash[0] == initial_hash[1] == metadata[0]["state_sha256"] == metadata[1]["state_sha256"],
        "initial_rng_equal": metadata[0]["rng"] == metadata[1]["rng"],
        "initial_metadata_equal": metadata[0] == metadata[1],
        "training_begin_rng_equal": read_json(first / "training_begin_rng.json") == read_json(second / "training_begin_rng.json"),
        "patient_order_equal": [row["train_patient_order_sha256"] for row in rows[0]] ==
                               [row["train_patient_order_sha256"] for row in rows[1]],
        "epoch_metrics_equal": rows[0] == rows[1],
        "best_epoch_equal": summaries[0]["best_epoch"] == summaries[1]["best_epoch"] == best[0]["epoch"] == best[1]["epoch"],
        "best_validation_mAP_equal": summaries[0]["best_validation_mAP"] == summaries[1]["best_validation_mAP"] ==
                                     best[0]["validation_mAP"] == best[1]["validation_mAP"],
        "best_model_state_equal": best_hash[0] == best_hash[1],
        "val_predictions_equal": predictions_equal["val"], "test_predictions_equal": predictions_equal["test"],
        "strict_environment_recorded": all(environment.get("deterministic_algorithms") == "enabled_strict"
                                           and environment.get("deterministic_enabled") is True
                                           and environment.get("deterministic_warn_only") is False for environment in environments),
        "environment_equal": environments[0] == environments[1],
        "model_configuration_equal": read_json(first / "model_config.json") == read_json(second / "model_config.json"),
    }
    require(checks == gate["checks"] and all(checks.values()), f"Strict A/A checks do not reproduce for seed {seed}")
    require(all(read_json(path / "prepared_sha256.json") == manifest["prepared_sha256"] for path in (first, second)),
            "Strict A/A inputs differ from the freeze")
    for path in (first, second):
        provenance = read_json(path / "strict_provenance.json")
        require(provenance.get("frozen_manifest_sha256") == gate["frozen_manifest_sha256"]
                and provenance.get("source_sha256") == manifest["source_sha256"]
                and provenance.get("strict_protocol_sha256") == manifest["strict_protocol_sha256"],
                "Strict A/A provenance differs from the freeze")


def check_frozen(root, audit, strict):
    """Old implementation/input gates remain required in the strict rerun."""
    legacy = root / "artifacts/experimental_audit"
    base = read_json(legacy / "frozen_manifest.json")
    require(tuple(base.get("seeds", [])) == SEEDS, "Frozen seed list changed")
    require(base.get("legacy_model_ast_matches_previous_export") is True,
            "Legacy model AST gate was not frozen as passed")
    prepared = root / "artifacts/optimization/prepared_l512"
    require(resolve_path(root, base["prepared_dir"]) == prepared.resolve(),
            "Frozen cache is not the declared legacy prepared_l512")
    hashes = base["prepared_sha256"]
    require({"train.npz", "val.npz", "test.npz", "metadata.json", "vocabulary.json",
             "preprocessing.json", "audit.json"}.issubset(hashes), "Incomplete frozen input hashes")
    for name, expected in hashes.items():
        require(file_sha(prepared / name) == expected, f"Frozen input changed: {name}")
    require(file_sha(root / "docs/experimental_protocol.md") == base["protocol_sha256"],
            "Original frozen protocol changed")
    for seed in SEEDS:
        reference = root / f"artifacts/optimization/runs/no-pretrain-l512-seed{seed}"
        for name, expected in base["references"][str(seed)].items():
            require(file_sha(reference / name) == expected, f"Legacy reference changed: {reference / name}")
    reload_gate = checked_gate(legacy / "gate_a_reload.json", SEEDS, "runs")
    for row in reload_gate["runs"]:
        require(row.get("passed") is True and abs(row["max_probability_difference"]) <= 1e-6
                and abs(row["mAP_difference"]) <= 1e-12, "Saved-checkpoint reload tolerance failed")
    bridge_gate = checked_gate(legacy / "gate_b_equivalence.json", SEEDS, "seeds")
    require(str(bridge_gate.get("device")).lower() == "cpu" and bridge_gate.get("patients") == 4,
            "Gate B did not use the frozen CPU/four-patient check")
    for row in bridge_gate["seeds"]:
        require(row.get("adapter_logit_loss_grad_and_step_identical") is True
                and row.get("four_cell_initial_rng_identical") is True
                and row["position_capacity_max_error"] <= 1e-6, "Gate B component failed")
    gate_sources = [legacy / "frozen_manifest.json", legacy / "gate_a_reload.json",
                    legacy / "gate_b_equivalence.json"]
    replay_gates = {}
    manifest = base
    if strict:
        manifest = read_json(audit / "frozen_manifest.json")
        require(tuple(manifest.get("seeds", [])) == SEEDS, "Strict frozen seeds changed")
        require(manifest.get("prepared_sha256") == hashes, "Strict and original input hashes differ")
        require(manifest.get("deterministic_mode") == "strict", "Strict manifest does not declare strict execution")
        require(resolve_path(root, manifest["strict_protocol_path"]) ==
                (root / "docs/experimental_protocol_strict.md").resolve(), "Wrong strict protocol path")
        require(file_sha(root / "docs/experimental_protocol_strict.md") == manifest["strict_protocol_sha256"],
                "Strict frozen protocol changed")
        require(resolve_path(root, manifest["original_frozen_manifest_path"]) ==
                (legacy / "frozen_manifest.json").resolve()
                and file_sha(legacy / "frozen_manifest.json") == manifest["original_frozen_manifest_sha256"]
                and manifest["original_protocol_sha256"] == base["protocol_sha256"],
                "Strict audit does not reference the unchanged original freeze")
        for name in ("gate_a_reload.json", "gate_b_equivalence.json"):
            require(manifest["upstream_gates_sha256"][name] == file_sha(legacy / name),
                    f"Strict upstream gate changed: {name}")
        require(isinstance(manifest.get("source_sha256"), dict) and manifest["source_sha256"],
                "Strict manifest lacks a frozen source hash mapping")
        for name, expected in manifest["source_sha256"].items():
            require(file_sha(resolve_path(root, name)) == expected, f"Strict frozen source changed: {name}")
        require(manifest.get("auxiliary_inputs_sha256"), "Strict manifest lacks auxiliary input hashes")
        for name, expected in manifest["auxiliary_inputs_sha256"].items():
            require(file_sha(resolve_path(root, name)) == expected, f"Strict auxiliary input changed: {name}")
        gate_sources.append(audit / "frozen_manifest.json")
        for seed in SEEDS:
            path = audit / f"gate_a_strict_seed{seed}.json"
            gate = checked_gate(path)
            require(gate.get("seed") == seed and gate.get("deterministic_mode") == "strict",
                    f"Wrong strict Gate A seed/mode: {path}")
            required_checks = {
                "configuration_equal_except_name_output", "initial_state_equal", "initial_rng_equal",
                "training_begin_rng_equal", "patient_order_equal", "epoch_metrics_equal", "best_epoch_equal",
                "best_model_state_equal", "val_predictions_equal", "test_predictions_equal", "strict_environment_recorded",
            }
            require(required_checks.issubset(gate.get("checks", {}))
                    and all(value is True for value in gate["checks"].values()), f"Strict A/A component failed: {path}")
            for key in ("baseline_run_dir", "repeat_run_dir"):
                require(resolve_path(root, gate[key]).is_dir(), f"Missing strict A/A run: {path}, {key}")
            recheck_strict_pair(root, audit, prepared, manifest, gate)
            replay_gates[seed] = gate
            gate_sources.append(path)
    else:
        for seed in SEEDS:
            path = audit / f"gate_a_replay_seed{seed}.json"
            gate = checked_gate(path)
            require(gate.get("seed") == seed and abs(gate["best_mAP_difference"]) <= .002
                    and abs(gate["max_fixed_epoch_mAP_difference"]) <= .005,
                    f"Legacy A/A alarm threshold failed: {path}")
            replay_gates[seed] = gate
            gate_sources.append(path)
    return prepared, hashes, manifest, replay_gates, {
        "passed": True, "strict": strict, "saved_checkpoint_reload": reload_gate,
        "implementation_equivalence": bridge_gate, "training_reproduction": replay_gates,
        "source_files_sha256": {str(path.relative_to(root)): file_sha(path) for path in gate_sources},
    }


def config_signature(config, root):
    result = {key: value for key, value in config.items() if key not in NON_TRAINING_FIELDS}
    result["prepared_dir"] = str(resolve_path(root, result["prepared_dir"]))
    return result


def read_predictions(path, patient_ids, codes):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        require(next(reader) == ["patient_id", *codes], f"Prediction column order mismatch: {path}")
        rows = list(reader)
    require(all(len(row) == len(codes) + 1 for row in rows), f"Malformed prediction row: {path}")
    require([row[0] for row in rows] == patient_ids, f"Prediction patient order mismatch: {path}")
    scores = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=np.float64)
    validate_probability_matrix(scores, n_patients=len(patient_ids), n_targets=len(codes))
    return scores


def load_groups(root, ids):
    path = root / "artifacts/optimization/history_patient_groups.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "val"]
    require(len(rows) == 365 and len({row["patient_id"] for row in rows}) == 365,
            "Frozen history groups have duplicate/missing validation patients")
    mapping = {row["patient_id"]: row for row in rows}
    require(set(mapping) == set(ids), "Frozen groups do not cover the official validation set")
    for row in rows:
        require(row["history_group_256"] == ("long" if int(row["full_event_count"]) > 255 else "short"),
                "Frozen history-group definition changed")
    groups = {name: np.array([mapping[patient]["history_group_256"] == name for patient in ids])
              for name in ("long", "short")}
    require(int(groups["long"].sum()) == 133 and int(groups["short"].sum()) == 232,
            "Expected fixed 133 long/232 short validation patients")
    return groups, {"path": str(path.relative_to(root)), "sha256": file_sha(path),
                    "definition": "Full deduplicated pre-anchor raw events >255; fixed long=133/short=232"}


def load_runs(root, audit, frozen_hashes, ids, codes, labels, prior, strict, replay_gates, manifest):
    predictions = {cell: {} for cell in CELLS}
    runs, seed_controls = {}, {}
    reference_signatures = {}
    environments = []
    for seed in SEEDS:
        baseline_config = read_json(root / f"artifacts/optimization/runs/no-pretrain-l512-seed{seed}/config.json")
        reference = config_signature(baseline_config, root)
        # Strict numerical execution is a declared new experimental condition.
        if strict:
            reference["deterministic_mode"] = "strict"
        reference_signatures[seed] = reference
        shared_states, rngs, orders = {}, {}, {}
        for cell in CELLS:
            folder = (resolve_path(root, replay_gates[seed]["baseline_run_dir"])
                      if strict and cell == "C0R0" else audit / f"runs/{cell}-seed{seed}")
            require(folder.is_relative_to(audit.resolve()), f"Run directory escaped the declared audit: {folder}")
            required = ["summary.json", "config.json", "initial.pt", "initial_state.json", "training_begin_rng.json",
                        "supervised_parameter_audit.json", "environment.json", "prepared_sha256.json", "history.jsonl",
                        "model_config.json", "best.pt", "validation_metrics.json", "val_predictions.csv"]
            require(all((folder / name).is_file() for name in required), f"Incomplete fitted run: {folder}")
            if strict:
                provenance = read_json(folder / "strict_provenance.json")
                require(provenance.get("frozen_manifest_sha256") == file_sha(audit / "frozen_manifest.json")
                        and provenance.get("source_sha256") == manifest["source_sha256"]
                        and provenance.get("strict_protocol_sha256") == manifest["strict_protocol_sha256"],
                        f"Strict run source/protocol provenance differs: {folder}")
                required.append("strict_provenance.json")
            config = read_json(folder / "config.json")
            require(config_signature(config, root) == reference, f"Undeclared training/config difference: {folder}")
            expected_attention, expected_readout = TREATMENTS[cell]
            require(config.get("attention_mode", "causal") == expected_attention
                    and config.get("readout_mode", "legacy") == expected_readout,
                    f"Cell treatment mismatch: {folder}")
            require(config["model_type"] == ("transformer" if cell == "C0R0" else "experimental_bridge"),
                    f"Wrong model class: {folder}")
            require(config.get("audit_diagnostics") is True, f"Missing audit instrumentation: {folder}")
            require(resolve_path(root, config["validation_groups_csv"]) ==
                    (root / "artifacts/optimization/history_patient_groups.csv").resolve(), "Changed subgroup source")
            require(read_json(folder / "prepared_sha256.json") == frozen_hashes, f"Run input hashes differ: {folder}")
            for name in ("metadata.json", "preprocessing.json", "vocabulary.json", "audit.json"):
                require(file_sha(folder / name) == frozen_hashes[name], f"Copied input metadata differs: {folder / name}")
            initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
            state = initial["model_state_dict"]
            model_config = read_json(folder / "model_config.json")
            require(initial.get("model_config") == model_config, f"Initial model configuration mismatch: {folder}")
            require(model_config.get("model_type") == config["model_type"]
                    and model_config.get("attention_mode", "causal") == expected_attention
                    and model_config.get("readout_mode", "legacy") == expected_readout,
                    f"Initialized model treatment differs from run configuration: {folder}")
            width, static_width = model_config["d_model"], model_config["n_static"]
            expected_head_width = width + static_width if expected_readout == "legacy" else width
            require(tuple(state["classification_head.weight"].shape) == (len(codes), expected_head_width),
                    f"Readout/head shape does not implement the declared treatment: {folder}")
            if expected_readout == "legacy":
                require(not any(name.startswith("readout.") for name in state), f"Unexpected new readout in R0: {folder}")
            else:
                require(tuple(state["readout.0.weight"].shape) == (width, 2 * width + static_width),
                        f"Missing/malformed pooled readout in R1: {folder}")
            initial_meta = read_json(folder / "initial_state.json")
            require(initial["seed"] == seed and initial_meta["state_sha256"] == state_sha(state),
                    f"Initial checkpoint fingerprint/seed mismatch: {folder}")
            common = {name: value for name, value in state.items()
                      if not name.startswith(("classification_head.", "readout."))}
            require(common, f"Missing common initial backbone: {folder}")
            shared_states[cell] = state_sha(common)
            rngs[cell] = read_json(folder / "training_begin_rng.json")
            require(rngs[cell] and all(isinstance(value, str) and len(value) == 64 for value in rngs[cell].values()),
                    f"Missing/invalid training-start RNG fingerprint: {folder}")
            environment = read_json(folder / "environment.json")
            require(environment["seed"] == seed, f"Environment seed mismatch: {folder}")
            require(environment.get("deterministic_algorithms") ==
                    ("enabled_strict" if strict else "enabled_warn_only"), f"Unexpected deterministic mode: {folder}")
            if strict:
                require(environment.get("deterministic_enabled") is True
                        and environment.get("deterministic_warn_only") is False,
                        f"Strict run used a permissive deterministic backend: {folder}")
            environments.append({key: value for key, value in environment.items() if key not in {"seed", "parameter_count"}})
            rows = history(folder / "history.jsonl")
            for row in rows:
                for key in ("train/loss", "val/loss", "val/mAP", "train_eval/BCE", "train_eval/mAP",
                            "gradient/norm_mean", "gradient/norm_max", "gradient/clip_fraction", "amp/skipped_steps"):
                    require(key in row and math.isfinite(row[key]), f"Missing/nonfinite diagnostic {key}: {folder}")
                require(0 <= row["gradient/clip_fraction"] <= 1 and row["amp/skipped_steps"] >= 0,
                        f"Invalid gradient/AMP diagnostics: {folder}")
                fingerprint = row.get("train_patient_order_sha256", "")
                require(isinstance(fingerprint, str) and len(fingerprint) == 64, f"Missing epoch order hash: {folder}")
            orders[cell] = {row["epoch"]: row["train_patient_order_sha256"] for row in rows}
            summary = read_json(folder / "summary.json")
            require(file_sha(folder / "best.pt") == summary["checkpoint_sha256"], f"Best checkpoint changed: {folder}")
            best = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            require(best["seed"] == seed and best["target_codes"] == codes
                    and best["prepared_sha256"] == frozen_hashes and best["config"] == config
                    and best["model_config"] == model_config,
                    f"Selected checkpoint provenance mismatch: {folder}")
            require(best["epoch"] == summary["best_epoch"] and 1 <= best["epoch"] <= len(rows),
                    f"Selected epoch inconsistent: {folder}")
            selected = rows[best["epoch"] - 1]
            require(abs(selected["val/mAP"] - summary["best_validation_mAP"]) < 1e-10
                    and abs(best["validation_mAP"] - summary["best_validation_mAP"]) < 1e-10,
                    f"Checkpoint/curve selection mismatch: {folder}")
            # Replicate the declared >1e-6 selection rule, including tie handling.
            selected_epoch, selected_score = None, -math.inf
            for row in rows:
                if row["val/mAP"] > selected_score + 1e-6:
                    selected_epoch, selected_score = row["epoch"], row["val/mAP"]
            require(selected_epoch == best["epoch"], f"Checkpoint was not selected by overall validation mAP: {folder}")
            scores = read_predictions(folder / "val_predictions.csv", ids, codes)
            require(not np.any(scores[prior] != 0), f"Previously diagnosed conditions were not zeroed: {folder}")
            recomputed = evaluate_predictions(labels, scores, codes)
            saved = read_json(folder / "validation_metrics.json")
            for metric in ("mAP", "macro_auroc", "brier", "micro_ap", "micro_auroc"):
                require(abs(recomputed[metric] - saved[metric]) <= 1e-8, f"Saved {metric} does not reproduce: {folder}")
            require(abs(recomputed["mAP"] - summary["best_validation_mAP"]) <= 1e-8, f"Prediction mAP mismatch: {folder}")
            parameter_audit = read_json(folder / "supervised_parameter_audit.json")
            with_grad = parameter_audit["with_gradient"]
            without_grad = parameter_audit["without_gradient"]
            actual_sizes = {name: value.numel() for name, value in state.items()}
            require(not set(with_grad) & set(without_grad) and {**with_grad, **without_grad} == actual_sizes,
                    f"Parameter audit does not partition initialized model parameters: {folder}")
            total = sum(actual_sizes.values())
            require(total == initial_meta["total_parameters"] == environment["parameter_count"],
                    f"Total parameter count mismatch: {folder}")
            positions = state["position_embedding.weight"]
            used_positions = int(initial_meta["input_shape"][1])
            require(used_positions == 512 and initial_meta["input_shape"][0] == 2791,
                    f"Unexpected legacy input shape: {folder}")
            position_unused = max(0, positions.shape[0] - used_positions) * positions.shape[1]
            token_unused = max(0, state["token_embedding.weight"].shape[0] -
                               int(initial_meta["observed_train_token_ids"])) * state["token_embedding.weight"].shape[1]
            key = f"{cell}-seed{seed}"
            runs[key] = {
                "cell": cell, "seed": seed, "folder": str(folder.relative_to(root)), "summary": summary,
                "history": rows, "parameters": {
                    "total": total, "gradient_bearing_tensor_elements": sum(with_grad.values()),
                    "without_gradient_tensor_elements": sum(without_grad.values()),
                    "unobserved_position_row_elements": position_unused,
                    "unobserved_token_row_elements": token_unused,
                    "observed_row_supervised_element_upper_bound": sum(with_grad.values()) - position_unused - token_unused,
                    "note": "First-step gradient-bearing tensors and observed-row upper bound are not effective model capacity.",
                    "first_step_audit": parameter_audit,
                },
                "diagnostics": {
                    "selected_epoch_eval_train_BCE": selected["train_eval/BCE"],
                    "selected_epoch_eval_val_BCE": selected["val/loss"],
                    "selected_epoch_val_minus_train_BCE": selected["val/loss"] - selected["train_eval/BCE"],
                    "maximum_preclip_gradient_norm": max(row["gradient/norm_max"] for row in rows),
                    "maximum_epoch_clip_fraction": max(row["gradient/clip_fraction"] for row in rows),
                    "total_amp_skipped_steps": sum(row["amp/skipped_steps"] for row in rows),
                    "loss_comparability": "train_eval/BCE and val/loss both use eval-mode at-risk BCE; train/loss is online updating/dropout BCE.",
                },
                "source_sha256": {name: file_sha(folder / name) for name in required},
            }
            predictions[cell][seed] = {"probabilities": scores, "patient_ids": ids, "target_codes": codes}
            del initial, state, common, best
        require(len(set(shared_states.values())) == 1, f"Common initial backbone differs across cells, seed {seed}")
        require(all(value == rngs["C0R0"] for value in rngs.values()), f"Training-start RNG differs, seed {seed}")
        overlap_checks = 0
        for epoch in sorted(set().union(*(set(order) for order in orders.values()))):
            present = [order[epoch] for order in orders.values() if epoch in order]
            require(len(set(present)) == 1, f"Patient batch order differs, seed {seed}, epoch {epoch}")
            overlap_checks += int(len(present) > 1)
        seed_controls[seed] = {"common_initial_backbone_sha256": shared_states["C0R0"],
                               "training_begin_rng": rngs["C0R0"], "overlapping_epochs_checked": overlap_checks,
                               "rng_scope": "Equal training-start states; subsequent stochastic consumption may differ by graph."}
    require(all(value == environments[0] for value in environments), "Runtime environments differ between factorial runs")
    across_seed = [{key: value for key, value in signature.items() if key != "seed"}
                   for signature in reference_signatures.values()]
    require(all(value == across_seed[0] for value in across_seed), "Non-seed training settings differ between reference seeds")
    return predictions, runs, {"passed": True, "by_seed": seed_controls, "environment": environments[0],
                               "training_settings": across_seed[0], "permitted_fields": sorted(NON_TRAINING_FIELDS)}


def save_figure(fig, path, note):
    fig.text(.01, .012, note, ha="left", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_plots(root, audit, results, runs, strict, replay_gates):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
    for column, seed in enumerate(SEEDS):
        if strict:
            reference = history(resolve_path(root, replay_gates[seed]["repeat_run_dir"]) / "history.jsonl")
            label = "Strict repeat B"
        else:
            reference = history(root / f"artifacts/optimization/runs/no-pretrain-l512-seed{seed}/history.jsonl")
            label = "Saved legacy run"
        replay = runs[f"C0R0-seed{seed}"]["history"]
        ax = axes[0, column]
        ax.plot([row["epoch"] for row in reference], [row["val/mAP"] for row in reference], label=label)
        ax.plot([row["epoch"] for row in replay], [row["val/mAP"] for row in replay], "--", label="Factorial C0R0 repeat A")
        ax.set(title=f"Seed {seed}: A/A reproduction", ylabel="Overall validation mAP")
        ax.legend(fontsize=8)
        left = {row["epoch"]: row["val/mAP"] for row in reference}
        epochs = [row["epoch"] for row in replay if row["epoch"] in left]
        delta = [replay[epoch - 1]["val/mAP"] - left[epoch] for epoch in epochs]
        ax = axes[1, column]
        ax.plot(epochs, delta, marker=".")
        ax.axhline(0, color="black", linewidth=.8)
        if not strict:
            ax.axhline(.005, color="grey", linestyle=":")
            ax.axhline(-.005, color="grey", linestyle=":")
        ax.set(xlabel="Finetuning epoch", ylabel="mAP: repeat A minus reference")
    fig.suptitle("A/A training reproduction: fixed seeds, complete overall validation cohort (n=365)")
    save_figure(fig, audit / "AA_reproduction.png", SOURCE_NOTE + " A/A tolerances are audit alarms, not confidence intervals.")

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, cell in zip(axes.flat, CELLS):
        for index, seed in enumerate(SEEDS):
            rows = runs[f"{cell}-seed{seed}"]["history"]
            epochs = [row["epoch"] for row in rows]
            ax.plot(epochs, [row["train_eval/BCE"] for row in rows], color=colors[index], label=f"Seed {seed}: train eval")
            ax.plot(epochs, [row["val/loss"] for row in rows], "--", color=colors[index], label=f"Seed {seed}: validation")
            best_epoch = runs[f"{cell}-seed{seed}"]["summary"]["best_epoch"]
            ax.scatter([best_epoch], [rows[best_epoch - 1]["val/loss"]], color=colors[index], s=25)
        ax.set(title=f"{cell}: C={TREATMENTS[cell][0]}, R={TREATMENTS[cell][1]}",
               xlabel="Finetuning epoch", ylabel="At-risk BCE (eval mode, lower is better)")
        ax.legend(fontsize=8)
    fig.suptitle("Comparable train/validation losses; dots mark checkpoints selected by overall validation mAP")
    save_figure(fig, audit / "train_val_BCE.png", SOURCE_NOTE + " Online train/loss is excluded because its model and dropout change within the epoch.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (scope, cohort) in zip(axes, [("Overall", results["overall"]), *[(name.title(), results["groups"][name]) for name in ("long", "short")]]):
        for index, cell in enumerate(CELLS):
            summary = cohort["cells"][cell]["across_seeds"]["mAP"]
            ax.errorbar(index, summary["mean"], yerr=summary["sample_std"], fmt="o", color=colors[index], capsize=5)
            values = [entry["metrics"]["mAP"] for entry in cohort["cells"][cell]["per_seed"]]
            ax.scatter(np.array([index] * 3) + np.array([-.08, 0, .08]), values, color=colors[index], alpha=.5, s=16)
        ax.set(title=f"{scope}, n={cohort['n_patients']}", ylabel="mAP (higher is better)", xticks=range(4), xticklabels=CELLS)
    fig.suptitle("Single-model validation mAP: mean +/- sample SD over all three seeds; small dots are individual seeds")
    save_figure(fig, audit / "factorial_metrics.png", SOURCE_NOTE + " Error bars are between-seed SD, not confidence intervals. C: attention; R: readout package.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    names = ("C_main", "R_main", "interaction")
    labels = ("C: bidirectional - causal\n(average over R)", "R: pooled MLP - legacy\n(average over C)", "C x R interaction\n(difference of differences)")
    for ax, (scope, cohort) in zip(axes, [("Overall", results["overall"]), *[(name.title(), results["groups"][name]) for name in ("long", "short")]]):
        for index, name in enumerate(names):
            contrast = cohort["contrasts"][name]
            interval = contrast["intervals"]["mAP"]
            if interval["ci_lower"] is not None:
                ax.hlines(index, interval["ci_lower"], interval["ci_upper"], color=colors[index], linewidth=2)
            ax.scatter([contrast["mean_single_model_contrast"]["mAP"]], [index], color=colors[index], zorder=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=f"{scope}, n={cohort['n_patients']}", xlabel="mAP contrast (positive increases mAP)", yticks=range(3), yticklabels=labels)
        ax.set_ylim(2.4, -.4)
    fig.suptitle(f"Factorial contrasts of per-seed metrics: 95% paired patient bootstrap intervals ({results['overall']['n_bootstrap']} resamples)")
    save_figure(fig, audit / "effects_mAP.png", "Fixed seeds/checkpoints; conditional patient sampling intervals, not seed SD. " + UNCERTAINTY_NOTE)


def analyze(root=ROOT, audit_dir=None, *, n_bootstrap=500, strict=False):
    root = Path(root).resolve()
    audit = resolve_path(root, audit_dir) if audit_dir else root / ("artifacts/experimental_audit_strict" if strict else "artifacts/experimental_audit")
    require(isinstance(n_bootstrap, int) and not isinstance(n_bootstrap, bool) and n_bootstrap >= 2,
            "At least two bootstrap resamples are required")
    prepared, frozen_hashes, manifest, replay_gates, gates = check_frozen(root, audit, strict)
    metadata = read_json(prepared / "metadata.json")
    codes = metadata["target_codes"]
    with np.load(prepared / "val.npz", allow_pickle=False) as arrays:
        ids = arrays["patient_ids"].astype(str).tolist()
        labels = arrays["labels"].copy()
        prior = arrays["prior_conditions"].copy()
    require(len(ids) == 365 and len(codes) == 40 and prior.dtype == np.bool_, "Unexpected frozen validation dimensions/prior dtype")
    require(not np.any(labels.astype(bool) & prior), "Future-positive labels overlap prior diagnoses")
    groups, group_source = load_groups(root, ids)
    predictions, runs, controls = load_runs(root, audit, frozen_hashes, ids, codes, labels, prior, strict, replay_gates, manifest)
    # All alignment/implementation/initialization/order controls pass before effects exist.
    results = factorial_analysis(labels, predictions, codes, groups=groups, patient_ids=ids,
                                 n_bootstrap=n_bootstrap, seed=2026, expected_seeds=SEEDS,
                                 prior_conditions=prior, require_alignment_metadata=True)
    results["selection"] = "One checkpoint per run, chosen only by overall validation mAP with >1e-6 improvement and patience=8. No seed/model selection and no ensemble."
    results["controls_passed"] = True
    results["strict_numerical_execution"] = strict
    results["inference_limits"] = UNCERTAINTY_NOTE
    existing_path = root / "artifacts/experimental_audit/existing_diagnostics.json"
    existing = read_json(existing_path)
    decomposition = {key: existing[key] for key in ("target_change_concentration", "net_mean_single_model_mAP_delta",
                                                    "established_evidence", "interpretation_limits") if key in existing}
    summary = {
        "schema_version": 1, "purpose": "Gate-controlled, single-model factorial validation analysis",
        "uses_probability_ensemble": False, "seeds": list(SEEDS), "strict": strict,
        "gates": gates, "factorial_controls": controls, "prepared_sha256": frozen_hashes,
        "groups": group_source, "runs": runs,
        "existing_diagnostics_reference": {"path": str(existing_path.relative_to(root)), "sha256": file_sha(existing_path),
                                           "scope": "Earlier architecture comparison, not the new factorial contrast", "decomposition": decomposition},
        "ranking_probability_tradeoff_reference": {"path": "artifacts/experimental_audit/ranking_probability_tradeoff.csv",
                                                    "sha256": file_sha(root / "artifacts/experimental_audit/ranking_probability_tradeoff.csv")},
        "interpretation": UNCERTAINTY_NOTE + " Effects describe this frozen factorial comparison; they are not automatic causal or clinical conclusions.",
    }
    for name, value in (("factorial_results.json", results), ("audit_summary.json", summary)):
        (audit / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["cohort", "n_patients", "cell", "seed", "selected_epoch", "mAP", "macro_auroc", "micro_ap", "micro_auroc", "brier", "at_risk_bce"]
    with (audit / "factorial_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, cohort in [("overall", results["overall"]), *results["groups"].items()]:
            for cell in CELLS:
                for entry in cohort["cells"][cell]["per_seed"]:
                    writer.writerow({"cohort": name, "n_patients": cohort["n_patients"], "cell": cell,
                                     "seed": entry["seed"], "selected_epoch": runs[f"{cell}-seed{entry['seed']}"]["summary"]["best_epoch"],
                                     **{metric: entry["metrics"][metric] for metric in columns[5:]}})
    make_plots(root, audit, results, runs, strict, replay_gates)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, help="Audit directory, relative to the project or absolute")
    parser.add_argument("--strict", action="store_true", help="Require strict rerun gates/protocol/source hashes; default to experimental_audit_strict")
    parser.add_argument("--bootstrap", type=int, default=500, help="Shared paired patient resamples (default: 500)")
    args = parser.parse_args()
    summary = analyze(audit_dir=args.audit_dir, n_bootstrap=args.bootstrap, strict=args.strict)
    print(json.dumps({"passed": True, "strict": summary["strict"], "seeds": summary["seeds"],
                      "outputs": ["factorial_results.json", "audit_summary.json", "factorial_scores.csv",
                                  "AA_reproduction.png", "train_val_BCE.png", "factorial_metrics.png", "effects_mAP.png"]}))


if __name__ == "__main__":
    main()
