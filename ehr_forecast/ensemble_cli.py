"""Portable manifest resolution and exact preparation recipes for ensembles."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .train import sha256_file


def _beneath(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Manifest paths must stay inside their declared root")
    return path


def load_members(manifest_path: Path, prepared_root: Path) -> list[dict]:
    manifest_path, prepared_root = Path(manifest_path).resolve(), Path(prepared_root).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = payload.get("members", [])
    if not members or payload.get("aggregation") != "equal_weight_probability_mean":
        raise ValueError("Manifest must declare a nonempty equal-weight ensemble")
    result = []
    for member in members:
        if not math.isclose(float(member["weight"]), 1 / len(members), rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("Only an equal weight for every declared member is supported")
        checkpoint = _beneath(manifest_path.parent, member["checkpoint"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != member["checkpoint_sha256"]:
            raise ValueError(f"Member checkpoint missing or SHA256 differs: {checkpoint}")
        result.append({"checkpoint": checkpoint,
                       "prepared_dir": _beneath(prepared_root, member["prepared_relative_path"]),
                       "recipe": member})
    return result


def _verify_encoder(member):
    for name, expected in member["recipe"]["encoder_sha256"].items():
        path = member["prepared_dir"] / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Prepared encoder mismatch: {path}")


def prepare_inputs(manifest_path: Path, data_dir: Path, output_root: Path) -> dict:
    from .data import prepare_dataset
    from .experiments_data import prepare_length_ablation, prepare_summary_dataset
    members = load_members(manifest_path, output_root)
    completed = set()
    for member in members:
        cache, recipe = member["prepared_dir"], member["recipe"]
        if cache in completed:
            _verify_encoder(member)
            continue
        if (cache / "train.npz").exists():
            if not all((cache / name).is_file() for name in ("val.npz", "test.npz")):
                raise ValueError(f"Existing cache is incomplete: {cache}")
            _verify_encoder(member)
            completed.add(cache)
            continue
        kwargs = dict(recipe["preparation_kwargs"])
        mode = recipe["preparation_mode"]
        if mode == "summary":
            prepare_summary_dataset(data_dir, cache, **kwargs)
        elif mode == "length":
            length = int(kwargs.pop("max_events"))
            if cache.name != f"prepared_l{length}":
                raise ValueError("Length cache path and recipe disagree")
            prepare_length_ablation(data_dir, cache.parent, lengths=(length,), **kwargs)
        elif mode == "original":
            prepare_dataset(data_dir, cache, **kwargs)
        else:
            raise ValueError(f"Unknown preparation mode: {mode}")
        _verify_encoder(member)
        completed.add(cache)
    return {"prepared_caches": [str(path) for path in sorted(completed)], "n_members": len(members)}
