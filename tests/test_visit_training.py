import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import ehr_forecast.train as training


def test_collator_trims_unit_and_event_padding():
    row = {"tokens": torch.tensor([[4, 5, 0, 0], [6, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int16),
           "time_days": torch.tensor([100., 30., 0., 0.]), "static": torch.zeros(2),
           "prior_conditions": torch.zeros(2, dtype=torch.bool)}
    result = training.collate_patients([row, row])
    assert result["tokens"].shape == (2, 2, 2)
    assert result["tokens"].dtype == torch.int16
    assert result["time_days"].shape == (2, 2)


@pytest.mark.parametrize("model_type", ["compressed_transformer", "hierarchical_transformer"])
def test_unit_training_group_evaluation_and_checkpoint_reload(tmp_path, monkeypatch, model_type):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    metadata = {"target_codes": ["001", "002"], "target_descriptions": ["A", "B"],
                "vocab_size": 16, "n_static": 2, "max_events": 4, "max_unit_events": 4,
                "input_layout": "event_units", "schema_version": 4}
    for filename, data in [("metadata.json", metadata), ("vocabulary.json", {"train_only": True}),
                           ("preprocessing.json", {"unit_rule": "synthetic dates"}), ("audit.json", {})]:
        (prepared / filename).write_text(json.dumps(data))
    for split, count in [("train", 8), ("val", 4), ("test", 4)]:
        tokens = np.zeros((count, 4, 4), dtype=np.int16)
        tokens[:, 0, :2] = [4, 5]
        tokens[:, 1, 0] = 6 + np.arange(count) % 2
        times = np.tile(np.array([100, 30, 0, 0], dtype=np.float32), (count, 1))
        arrays = dict(patient_ids=np.array([f"{split}_{index}" for index in range(count)]),
                      tokens=tokens, time_days=times, static=np.ones((count, 2), np.float32),
                      prior_conditions=np.zeros((count, 2), bool))
        if split != "test":
            arrays["labels"] = np.column_stack([np.arange(count) % 2, np.arange(count) // 2 % 2]).astype(np.float32)
        np.savez_compressed(prepared / f"{split}.npz", **arrays)
    groups = tmp_path / "groups.csv"
    with groups.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "split", "history_group_256"])
        writer.writerows([[f"val_{index}", "val", "long" if index < 2 else "short"] for index in range(4)])
    output = tmp_path / "run"
    config = dict(prepared_dir=str(prepared), output_dir=str(output), model_type=model_type,
                  d_model=8, n_heads=2, n_layers=2, local_layers=1, global_layers=1,
                  window_size=2, max_unit_events=4, dropout=0., batch_size=4,
                  epochs=1, pretrain_epochs=0, learning_rate=.001, seed=17, device="cpu",
                  amp=False, wandb_mode="disabled", validation_groups_csv=str(groups))
    monkeypatch.setattr(training, "plot_curves", lambda path: None)
    result = training.train(config)
    assert result["best_epoch"] == 1
    groups_result = json.loads((output / "group_validation_metrics.json").read_text())
    assert groups_result["long"]["n_patients"] == 2
    assert groups_result["short"]["n_patients"] == 2
    history = json.loads((output / "history.jsonl").read_text().splitlines()[0])
    assert "val_long/mAP" in history
    reloaded = tmp_path / "reloaded.csv"
    training.predict(output / "best.pt", prepared, reloaded, device="cpu", split="val")
    assert reloaded.read_bytes() == (output / "val_predictions.csv").read_bytes()


def test_validation_groups_cannot_silently_omit_patients(tmp_path):
    path = tmp_path / "groups.csv"
    path.write_text("patient_id,split,history_group_256\na,val,long\n")
    with pytest.raises(ValueError, match="exactly the official"):
        training.validation_group_masks(str(path), ["a", "b"])
