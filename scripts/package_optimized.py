"""Package the selected complete-seed ensemble and reproduce its predictions."""

from pathlib import Path
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ehr_forecast.ensemble import export_ensemble, predict_ensemble
from ehr_forecast.experiment_analysis import average_validation_predictions, load_aligned_predictions
from ehr_forecast.submission import validate_predictions
from ehr_forecast.train import write_json


def main():
    root = ROOT / "artifacts/optimization"
    analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    if len(analysis["families"]) != 10 or not analysis["all_controlled_lengths_complete"]:
        raise ValueError("Complete the declared experiment families before packaging")
    selected = analysis["selected_recipe"]
    members, validation_means = [], []
    for family in selected["families"]:
        runs = analysis["families"][family]["per_seed"]
        if sorted(run["seed"] for run in runs) != [17, 42, 73]:
            raise ValueError("Every selected family must include all declared seeds")
        directories = [Path(run["run_dir"]) for run in runs]
        prepared = ROOT / json.loads((directories[0] / "config.json").read_text())["prepared_dir"]
        validation_means.append(average_validation_predictions(directories, prepared, expected_seeds=[17, 42, 73]))
        members.extend({"checkpoint": directory / "best.pt", "prepared_dir": prepared} for directory in directories)
    bundle = ROOT / "artifacts/optimized_model_bundle"
    exported = export_ensemble(members, bundle)
    if abs(exported["validation_mAP"] - selected["metrics"]["mAP"]) > 1e-9:
        raise ValueError("Export metrics disagree with selected validation recipe")
    inference = predict_ensemble(members, root / "predictions_optimized.csv")
    predict_ensemble(members, root / "validation_predictions_optimized.csv", split="val")
    metadata = json.loads((members[0]["prepared_dir"] / "metadata.json").read_text())
    with np.load(members[0]["prepared_dir"] / "val.npz", allow_pickle=False) as arrays:
        ids = arrays["patient_ids"].astype(str).tolist()
    reloaded = load_aligned_predictions(root / "validation_predictions_optimized.csv", ids, metadata["target_codes"])
    difference = float(np.max(np.abs(reloaded - np.mean(validation_means, axis=0))))
    if difference > 1e-7:
        raise ValueError("Reloaded ensemble predictions do not reproduce saved validation scores")
    backup = root / "predictions_before_optimization.csv"
    if not backup.exists() and (ROOT / "predictions.csv").exists():
        shutil.copy2(ROOT / "predictions.csv", backup)
    temporary = ROOT / "predictions.optimized.tmp"
    shutil.copy2(root / "predictions_optimized.csv", temporary)
    temporary.replace(ROOT / "predictions.csv")
    with np.load(members[0]["prepared_dir"] / "test.npz", allow_pickle=False) as arrays:
        test_ids = arrays["patient_ids"].astype(str).tolist()
    checked = validate_predictions(ROOT / "predictions.csv", metadata["target_codes"], test_ids)
    result = {"selected": selected, "export": exported, "inference": inference,
              "submission_validation": checked, "reloaded_validation_max_absolute_difference": difference,
              "previous_predictions_backup": str(backup), "test_metrics": None}
    write_json(root / "selected_model.json", result)
    print(json.dumps({"selected": selected["name"], "families": selected["families"],
                      "mAP": selected["metrics"]["mAP"], "macro_auroc": selected["metrics"]["macro_auroc"],
                      "n_members": len(members), "val_reproduction_max_diff": difference,
                      "submission": checked}, indent=2))


if __name__ == "__main__":
    main()
