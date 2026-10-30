"""Reload all eighteen exported single models through the original inference path."""
from pathlib import Path
import csv
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_group_ranking as study
from ranking_study.trainer import file_sha, read_json, require, write_json
from ehr_forecast.models import build_model
from ehr_forecast.train import PatientDataset, make_loader, predict_probabilities, seed_everything
from ehr_forecast.metrics import evaluate_predictions


def saved_predictions(path, ids, codes):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    require(rows[0] == ["patient_id", *codes] and [r[0] for r in rows[1:]] == ids,
            f"Saved prediction order changed: {path}")
    return np.asarray([r[1:] for r in rows[1:]], float)


def main():
    study.verify()
    require(read_json(study.STUDY / "reproduction_gates.json")["passed"], "Reproduction must pass first")
    prepared = ROOT / "artifacts/optimization/prepared_l512"
    metadata = read_json(prepared / "metadata.json")
    codes = metadata["target_codes"]
    data = {split: PatientDataset(prepared / f"{split}.npz") for split in ("val", "test")}
    device = torch.device("cuda")
    require(torch.cuda.is_available(), "The declared export verification uses the same GPU")
    seed_everything(17, "strict")
    results = []
    for method in study.METHODS:
        for seed in study.SEEDS:
            folder = study.STUDY / "runs" / f"{method}-seed{seed}"
            summary = read_json(folder / "summary.json")
            require(summary["checkpoint_sha256"] == file_sha(folder / "best.pt"), "Export changed")
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            require(checkpoint["target_codes"] == codes, "Checkpoint target order changed")
            model = build_model(checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            for split in (["val", "test"] if seed == 42 and method in {"bce", "listnet"} else ["val"]):
                dataset = data[split]
                probabilities = predict_probabilities(model, make_loader(dataset, 64), device)
                old = saved_predictions(folder / f"{split}_predictions.csv", dataset.patient_ids, codes)
                maximum = float(np.max(np.abs(old-probabilities)))
                score_delta = None
                if split == "val":
                    score = evaluate_predictions(dataset.arrays["labels"], probabilities, codes)["mAP"]
                    score_delta = score-read_json(folder / "validation_metrics.json")["mAP"]
                passed = maximum <= 1e-6 and (score_delta is None or abs(score_delta) <= 1e-10)
                results.append({"method": method, "seed": seed, "split": split, "patients": len(dataset),
                                "max_probability_difference": maximum, "mAP_difference": score_delta,
                                "passed": passed, "checkpoint_sha256": file_sha(folder / "best.pt")})
            del model
            torch.cuda.empty_cache()
    report = {"passed": all(r["passed"] for r in results), "runs": results,
              "source_sha256": {study.relative(Path(__file__)): file_sha(Path(__file__))},
              "frozen_manifest_sha256": file_sha(study.STUDY / "frozen_manifest.json"),
              "scope": "Full original encoder+exported classifier inference; no cached features used for prediction.",
              "validation_tolerance": "max probability1e-6; mAP1e-10"}
    path = study.STUDY / "model_reload_checks.json"
    if path.exists():
        require(read_json(path) == report, "Refusing to overwrite different reload evidence")
    else:
        write_json(path, report)
    print({"passed": report["passed"], "validation_models": 18, "extra_test_checks": 2,
           "max_probability_difference": max(r["max_probability_difference"] for r in results)})
    require(report["passed"], "Export reload failed; do not promote a cached-feature score")


if __name__ == "__main__":
    main()
