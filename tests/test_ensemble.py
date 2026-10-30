import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ehr_forecast import ensemble
from ehr_forecast.submission import write_predictions


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def members(tmp_path, monkeypatch):
    result = []
    for index, (ids, codes) in enumerate([(["p1", "p2"], ["a", "b"]), (["p2", "p1"], ["b", "a"])]):
        run = tmp_path / f"run_{index}"
        prepared = tmp_path / f"prepared_{index}"
        run.mkdir()
        prepared.mkdir()
        checkpoint = run / "best.pt"
        checkpoint.write_bytes(f"fake weights {index}".encode())
        dump(run / "config.json", {"seed": 42 + index, "model_type": "fake"})
        metadata = {"target_codes": codes, "max_events": 256, "n_static": 5, "n_value_bins": 8, "min_token_count": 3}
        for destination in (run, prepared):
            dump(destination / "metadata.json", metadata)
            dump(destination / "vocabulary.json", {"dummy": index})
            dump(destination / "preprocessing.json", {"schema_version": 2, "vocabulary_scope": "training", "dummy": index})
        # p1 has a=0,b=1 and p2 a=1,b=0; order and target columns both differ.
        labels = np.array([[0, 1], [1, 0]])
        for split in ("val", "test"):
            np.savez(prepared / f"{split}.npz", patient_ids=ids, **({"labels": labels} if split == "val" else {}))
        dump(run / "summary.json", {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                    "test_metrics": None, "test_labels_available": False})
        for name in ("model_config.json", "validation_metrics.json", "environment.json"):
            dump(run / name, {})
        probabilities = [[.1, .9], [.8, .2]] if index == 0 else [[.4, .6], [.7, .3]]
        write_predictions(run / "val_predictions.csv", ids, probabilities, codes)
        result.append({"checkpoint": checkpoint, "prepared_dir": prepared})

    def fake_predict(checkpoint, prepared_dir, output_path, device, split):
        index = int(checkpoint.parent.name[-1])
        own = json.loads((prepared_dir / "metadata.json").read_text())
        ids = ["p2", "p1"] if index == 0 else ["p1", "p2"]
        # Even inference row order differs from its own prepared cache.
        probabilities = [[.8, .2], [.1, .9]] if index == 0 else [[.7, .3], [.4, .6]]
        write_predictions(output_path, ids, probabilities, own["target_codes"])

    monkeypatch.setattr(ensemble, "_predict_member", fake_predict)
    return result


def test_equal_average_aligns_patient_and_target_order(members, tmp_path):
    output = tmp_path / "predictions.csv"
    result = ensemble.predict_ensemble(members, output, device="cpu")
    assert result["n_members"] == 2
    assert result["weight_per_member"] == .5
    scores = ensemble._read_aligned_csv(output, ["a", "b"], ["p1", "p2"], ["a", "b"])
    np.testing.assert_allclose(scores, [[.2, .8], [.7, .3]])


def test_existing_output_is_preserved(members, tmp_path):
    output = tmp_path / "predictions.csv"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        ensemble.predict_ensemble(members, output)
    assert output.read_text() == "existing"


def test_mismatched_targets_rejected_before_inference(members, tmp_path):
    metadata = members[1]["prepared_dir"] / "metadata.json"
    dump(metadata, {"target_codes": ["a", "c"], "max_events": 256})
    with pytest.raises(ValueError, match="target CODE set"):
        ensemble.predict_ensemble(members, tmp_path / "predictions.csv")


def test_duplicate_member_and_patient_coverage_are_rejected(members, tmp_path):
    with pytest.raises(ValueError, match="Duplicate checkpoint"):
        ensemble.predict_ensemble([members[0], members[0]], tmp_path / "predictions.csv")
    np.savez(members[1]["prepared_dir"] / "test.npz", patient_ids=["p2", "other"])
    with pytest.raises(ValueError, match="patient coverage"):
        ensemble.predict_ensemble(members, tmp_path / "predictions.csv")


def test_member_encoder_failures_are_propagated_without_partial_output(members, tmp_path, monkeypatch):
    def failure(*args):
        raise ValueError("Checkpoint preprocessing does not match vocabulary.json")
    monkeypatch.setattr(ensemble, "_predict_member", failure)
    output = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="preprocessing"):
        ensemble.predict_ensemble(members, output)
    assert not output.exists()


def test_export_is_portable_hash_complete_and_excludes_patient_arrays(members, tmp_path):
    destination = tmp_path / "bundle"
    result = ensemble.export_ensemble(members, destination)
    assert result["n_members"] == 2
    assert not result["uploaded"]
    payload = json.loads((destination / "ensemble.json").read_text())
    assert payload["members"][0]["prepared_relative_path"] == "prepared_l256"
    assert payload["members"][1]["prepared_relative_path"] == "member_001/prepared_l256"
    assert all(not Path(member["checkpoint"]).is_absolute() for member in payload["members"])
    assert (destination / "ehr_forecast" / "ensemble.py").is_file()
    assert not list(destination.rglob("*.npz"))
    assert not list(destination.rglob("*.csv"))
    manifest = json.loads((destination / "manifest.json").read_text())
    files = {file.relative_to(destination).as_posix() for file in destination.rglob("*") if file.is_file()}
    assert set(manifest["files"]) == files - {"manifest.json"}
    for name, fingerprint in manifest["files"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == fingerprint["sha256"]
    with pytest.raises(FileExistsError):
        ensemble.export_ensemble(members, destination)


def test_export_rejects_checkpoint_not_selected_by_completed_run(members, tmp_path):
    members[0]["checkpoint"].write_bytes(b"changed checkpoint")
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="summary SHA256"):
        ensemble.export_ensemble(members, destination)
    assert not destination.exists()


def test_export_rejects_mismatched_frozen_encoder(members, tmp_path):
    dump(members[0]["prepared_dir"] / "vocabulary.json", {"changed": True})
    with pytest.raises(ValueError, match="Run encoder differs"):
        ensemble.export_ensemble(members, tmp_path / "bundle")
