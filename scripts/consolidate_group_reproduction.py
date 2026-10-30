"""Consolidate predeclared A/A gates and directly compare full model tensors."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_group_ranking as study
from ranking_study.trainer import file_sha, read_json, require, tensor_sha, write_json


def main():
    study.verify()
    paths = [study.STUDY / f"aa_{method}_seed17.json" for method in ("bce", "grpo_ap")]
    full = []
    for path in paths:
        gate = read_json(path)
        require(gate.get("passed") is True and all(gate["checks"].values()), "Raw A/A gate failed")
        for name, fingerprint in gate["artifact_sha256"].items():
            require(file_sha(ROOT / name) == fingerprint, f"A/A artifact changed: {name}")
        fingerprints = []
        for key in ("baseline_run_dir", "repeat_run_dir"):
            checkpoint = torch.load(ROOT / gate[key] / "best.pt", map_location="cpu", weights_only=True)
            fingerprints.append(tensor_sha(checkpoint["model_state_dict"]))
        full.append({"method": gate["method"], "seed": gate["seed"],
                     "full_model_tensor_equal": fingerprints[0] == fingerprints[1], "state_sha256": fingerprints})
    result = {"schema_version": 1, "passed": all(r["full_model_tensor_equal"] for r in full),
              "frozen_manifest_sha256": file_sha(study.STUDY / "frozen_manifest.json"),
              "gate_files": [study.relative(p) for p in paths],
              "gate_files_sha256": {study.relative(p): file_sha(p) for p in paths},
              "full_model_checks": full,
              "source_sha256": {study.relative(Path(__file__)): file_sha(Path(__file__))}}
    require(result["passed"], "Direct full-model A/A comparison failed")
    target = study.STUDY / "reproduction_gates.json"
    if target.exists():
        require(read_json(target) == result, "Refusing to replace different reproduction evidence")
    else:
        write_json(target, result)
    print({"passed": result["passed"], "full_model_checks": [(r["method"], r["full_model_tensor_equal"]) for r in full]})


if __name__ == "__main__":
    main()
