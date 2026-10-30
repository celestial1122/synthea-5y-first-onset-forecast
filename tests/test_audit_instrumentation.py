import json

import numpy as np
import pytest
import torch

import ehr_forecast.train as training


def test_strict_determinism_is_explicit_and_legacy_default_is_preserved():
    try:
        training.seed_everything(73, "strict")
        first = torch.rand(8)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        training.seed_everything(73, "strict")
        assert torch.equal(first, torch.rand(8))
        training.seed_everything(73)
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        with pytest.raises(ValueError, match="deterministic_mode"):
            training.seed_everything(73, "silently_relaxed")
    finally:
        training.seed_everything(73)


def test_diagnostics_do_not_change_cpu_training_or_rng(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    meta = dict(vocab_size=16, n_static=2, max_events=4, target_codes=["A", "B"],
                target_descriptions=["A", "B"])
    for name, value in [("metadata.json", meta), ("preprocessing.json", {}), ("vocabulary.json", {}), ("audit.json", {})]:
        (prepared / name).write_text(json.dumps(value))
    for split, count in [("train", 8), ("val", 4), ("test", 4)]:
        tokens = np.zeros((count, 4), np.int64)
        tokens[:, :3] = [2, 4, 5]
        tokens[:, 2] += np.arange(count) % 2
        arrays = dict(patient_ids=np.array([f"{split}_{index}" for index in range(count)]),
                      tokens=tokens, time_days=np.tile(np.array([0, 10, 5, 0], np.float32), (count, 1)),
                      static=np.ones((count, 2), np.float32), prior_conditions=np.zeros((count, 2), bool))
        if split != "test":
            arrays["labels"] = np.column_stack([np.arange(count) % 2, np.arange(count) // 2 % 2]).astype(np.float32)
        np.savez_compressed(prepared / f"{split}.npz", **arrays)
    monkeypatch.setattr(training, "plot_curves", lambda path: None)
    state_hashes, rngs, output_paths = [], [], []
    for enabled in (False, True):
        output = tmp_path / ("observed" if enabled else "normal")
        config = dict(prepared_dir=str(prepared), output_dir=str(output), model_type="transformer",
                      d_model=8, n_heads=2, n_layers=1, dropout=.1, batch_size=4,
                      seed=17, epochs=1, pretrain_epochs=0, device="cpu", amp=False,
                      learning_rate=.001, wandb_mode="disabled", audit_diagnostics=enabled)
        training.train(config)
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        state_hashes.append(training.state_fingerprint(checkpoint["model_state_dict"]))
        rngs.append(torch.get_rng_state().clone())
        output_paths.append(output)
    assert state_hashes[0] == state_hashes[1]
    assert torch.equal(rngs[0], rngs[1])
    assert (output_paths[0] / "val_predictions.csv").read_bytes() == (output_paths[1] / "val_predictions.csv").read_bytes()
    observed = output_paths[1]
    row = json.loads((observed / "history.jsonl").read_text().splitlines()[0])
    assert "train_eval/BCE" in row and "train_eval/mAP" in row
    assert row["amp/skipped_steps"] == 0
    assert 0 <= row["gradient/clip_fraction"] <= 1
    assert len(row["train_patient_order_sha256"]) == 64
    parameter_audit = json.loads((observed / "supervised_parameter_audit.json").read_text())
    assert "language_head.weight" in parameter_audit["without_gradient"]
