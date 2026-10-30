"""Tiny end-to-end checks for split isolation and deployable checkpoints.

These fixtures are synthetic and contain no released patients.  The module runs
one CPU training epoch once, then checks the actual saved/reloaded artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import torch

import ehr_forecast.train as training


TARGETS = ["001", "002"]


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepare_fixture(prepared: Path) -> Path:
    prepared.mkdir()
    _json(prepared / "metadata.json", {
        "target_codes": TARGETS,
        "target_descriptions": ["Synthetic condition A", "Synthetic condition B"],
        "vocab_size": 16,
        "n_static": 2,
        "max_events": 8,
        "split_counts": {"train": 8, "val": 4, "test": 4},
    })
    _json(prepared / "vocabulary.json", {
        "token_to_id": {"PAD": 0, "UNK": 1, "BOS": 2, "MASK": 3, "event": 4},
        "fit_split": "train",
    })
    _json(prepared / "preprocessing.json", {"fit_split": "train", "numeric_bins": {}})
    _json(prepared / "audit.json", {"synthetic_test_fixture": True})
    for split, count, marker_start in (("train", 8, 4), ("val", 4, 8), ("test", 4, 12)):
        tokens = np.zeros((count, 8), dtype=np.int64)
        tokens[:, 0] = 2
        # Split-specific first events let us observe exactly which split gets
        # gradient-enabled forwards without altering the training implementation.
        tokens[:, 1] = marker_start + np.arange(count) % 4
        tokens[:, 2:4] = [5, 6]
        times = np.zeros_like(tokens, dtype=np.float32)
        times[:, 1:4] = [100, 40, 5]
        prior = np.zeros((count, 2), dtype=bool)
        prior[0, 0] = True
        prior[1, 1] = True
        arrays = {
            "patient_ids": np.array([f"{split}_{index}" for index in range(count)]),
            "tokens": tokens,
            "time_days": times,
            "static": np.column_stack((np.linspace(-1, 1, count), np.arange(count) % 2)).astype(np.float32),
            "prior_conditions": prior,
        }
        if split != "test":
            arrays["labels"] = np.column_stack((np.arange(count) % 2, np.arange(count) // 2 % 2)).astype(np.float32)
        np.savez_compressed(prepared / f"{split}.npz", **arrays)
    return prepared


def _config(prepared: Path, output: Path) -> dict:
    return {
        "prepared_dir": str(prepared),
        "output_dir": str(output),
        "model_type": "bag",
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.0,
        "epochs": 1,
        "pretrain_epochs": 0,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "device": "cpu",
        "amp": False,
        "wandb_mode": "disabled",
        "seed": 19,
    }


def _rewrite_arrays(path: Path, mutator) -> None:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    mutator(arrays)
    np.savez_compressed(path, **arrays)


def _read_scores(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["patient_id", *TARGETS]
        rows = list(reader)
    return [row["patient_id"] for row in rows], np.array([[float(row[code]) for code in TARGETS] for row in rows])


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("training-integration")
    prepared = _prepare_fixture(root / "prepared")
    output = root / "run"
    config = _config(prepared, output)
    forwards = []
    original_logits = training.model_logits

    def observe_forward(model, batch):
        forwards.append({
            "markers": batch["tokens"][:, 1].detach().cpu().numpy().tolist(),
            "training": model.training,
            "grad_enabled": torch.is_grad_enabled(),
        })
        return original_logits(model, batch)

    with patch.object(training, "plot_curves"), patch.object(training, "model_logits", side_effect=observe_forward):
        summary = training.train(config)
    return {"prepared": prepared, "output": output, "config": config, "summary": summary, "forwards": forwards}


@pytest.fixture
def fresh_prepared(tmp_path):
    return _prepare_fixture(tmp_path / "prepared")


def test_training_keeps_validation_and_test_out_of_gradients(trained_run):
    forwards = trained_run["forwards"]
    assert any(event["training"] and event["grad_enabled"] for event in forwards)
    for event in forwards:
        if event["training"] or event["grad_enabled"]:
            assert all(4 <= marker < 8 for marker in event["markers"])
        if any(marker >= 8 for marker in event["markers"]):
            assert not event["training"]
            assert not event["grad_enabled"]
    assert any(any(8 <= marker < 12 for marker in event["markers"]) for event in forwards)
    assert any(any(marker >= 12 for marker in event["markers"]) for event in forwards)
    with np.load(trained_run["prepared"] / "test.npz", allow_pickle=False) as arrays:
        assert "labels" not in arrays.files
    assert trained_run["summary"]["test_labels_available"] is False
    assert trained_run["summary"]["test_metrics"] is None


@pytest.mark.parametrize("split", ["val", "test"])
def test_checkpoint_reload_reproduces_probabilities_and_prior_mask(trained_run, tmp_path, split):
    output = trained_run["output"]
    checkpoint = output / "best.pt"
    assert checkpoint.is_file()
    restored = tmp_path / f"restored_{split}.csv"
    result = training.predict(checkpoint, trained_run["prepared"], restored, device="cpu", split=split)
    original_ids, original = _read_scores(output / f"{split}_predictions.csv")
    restored_ids, scores = _read_scores(restored)
    assert result["patients"] == len(restored_ids) == 4
    assert result["conditions"] == 2
    assert restored_ids == original_ids
    np.testing.assert_allclose(scores, original, atol=1e-7, rtol=1e-6)
    with np.load(trained_run["prepared"] / f"{split}.npz", allow_pickle=False) as arrays:
        prior = arrays["prior_conditions"]
    assert np.all(scores[prior] == 0.0)
    assert np.all(scores[~prior] > 0.0)
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))


def test_train_rejects_labels_on_test_before_writing_checkpoint(fresh_prepared, tmp_path):
    _rewrite_arrays(fresh_prepared / "test.npz", lambda arrays: arrays.update(labels=np.zeros((4, 2), dtype=np.float32)))
    output = tmp_path / "bad-run"
    with pytest.raises(ValueError, match="Test labels must not be present"):
        training.train(_config(fresh_prepared, output))
    assert not (output / "best.pt").exists()


@pytest.mark.parametrize("source,destination", [("train", "val"), ("train", "test"), ("val", "test")])
def test_train_rejects_any_patient_overlap(fresh_prepared, tmp_path, source, destination):
    def overlap(arrays):
        arrays["patient_ids"] = arrays["patient_ids"].astype("U20")
        arrays["patient_ids"][0] = f"{source}_0"

    _rewrite_arrays(fresh_prepared / f"{destination}.npz", overlap)
    with pytest.raises(ValueError, match="Patient overlap"):
        training.train(_config(fresh_prepared, tmp_path / "bad-run"))


def test_train_rejects_incident_label_for_preexisting_condition(fresh_prepared, tmp_path):
    def corrupt_label(arrays):
        arrays["labels"][0, 0] = 1

    _rewrite_arrays(fresh_prepared / "train.npz", corrupt_label)
    with pytest.raises(ValueError, match="Incident labels overlap prior diagnoses"):
        training.train(_config(fresh_prepared, tmp_path / "bad-run"))


@pytest.mark.parametrize("filename", ["vocabulary.json", "preprocessing.json"])
def test_predict_rejects_changed_preprocessing_hash(trained_run, tmp_path, filename):
    changed = tmp_path / "changed"
    shutil.copytree(trained_run["prepared"], changed)
    payload = json.loads((changed / filename).read_text(encoding="utf-8"))
    payload["unexpected_change"] = True
    _json(changed / filename, payload)
    with pytest.raises(ValueError, match="Checkpoint preprocessing does not match"):
        training.predict(trained_run["output"] / "best.pt", changed, tmp_path / "bad.csv", device="cpu")
    assert not (tmp_path / "bad.csv").exists()


def test_predict_rejects_reordered_target_columns(trained_run, tmp_path):
    changed = tmp_path / "changed"
    shutil.copytree(trained_run["prepared"], changed)
    metadata = json.loads((changed / "metadata.json").read_text(encoding="utf-8"))
    metadata["target_codes"].reverse()
    _json(changed / "metadata.json", metadata)
    with pytest.raises(ValueError, match="Checkpoint target order"):
        training.predict(trained_run["output"] / "best.pt", changed, tmp_path / "bad.csv", device="cpu")


def test_existing_training_run_is_never_overwritten(trained_run):
    checkpoint = trained_run["output"] / "best.pt"
    original_hash = training.sha256_file(checkpoint)
    with pytest.raises(FileExistsError, match="already contains a run"):
        training.train(trained_run["config"])
    assert training.sha256_file(checkpoint) == original_hash
