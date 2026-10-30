import hashlib
import json
from pathlib import Path

import pytest

from ehr_forecast.ensemble_cli import load_members, prepare_inputs


@pytest.fixture
def manifest(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint for path and hash checks")
    recipe = {
        "checkpoint": "model.pt", "weight": 1.0,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "prepared_relative_path": "prepared_summary", "preparation_mode": "summary",
        "preparation_kwargs": {"max_events": 256, "min_token_count": 3, "n_value_bins": 8},
        "encoder_sha256": {},
    }
    path = tmp_path / "ensemble.json"
    path.write_text(json.dumps({"aggregation": "equal_weight_probability_mean", "members": [recipe]}))
    return path


def test_relative_manifest_and_existing_encoder(manifest, tmp_path):
    root = tmp_path / "caches"
    cache = root / "prepared_summary"
    cache.mkdir(parents=True)
    payload = json.loads(manifest.read_text())
    for name in ("vocabulary.json", "preprocessing.json"):
        (cache / name).write_text("{}")
        payload["members"][0]["encoder_sha256"][name] = hashlib.sha256((cache / name).read_bytes()).hexdigest()
    for name in ("train.npz", "val.npz", "test.npz"):
        (cache / name).write_bytes(b"cache existence fixture")
    manifest.write_text(json.dumps(payload))
    assert load_members(manifest, root)[0]["prepared_dir"] == cache
    assert prepare_inputs(manifest, Path("unused_official_data"), root)["n_members"] == 1
    (cache / "preprocessing.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="encoder mismatch"):
        prepare_inputs(manifest, Path("unused_official_data"), root)


@pytest.mark.parametrize("field,value", [("checkpoint", "../outside.pt"), ("prepared_relative_path", "../outside")])
def test_manifest_cannot_escape_roots(manifest, tmp_path, field, value):
    payload = json.loads(manifest.read_text())
    payload["members"][0][field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inside their declared root"):
        load_members(manifest, tmp_path / "caches")


def test_manifest_rejects_changed_weights_or_checkpoint(manifest, tmp_path):
    payload = json.loads(manifest.read_text())
    payload["members"][0]["weight"] = .5
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="equal weight"):
        load_members(manifest, tmp_path / "caches")
    payload["members"][0]["weight"] = 1
    manifest.write_text(json.dumps(payload))
    (tmp_path / "model.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA256 differs"):
        load_members(manifest, tmp_path / "caches")
