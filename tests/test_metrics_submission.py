"""Metrics and artifact checks protect scoring semantics and patient coverage."""

import json

import numpy as np
import pytest

from ehr_forecast.metrics import evaluate_predictions
from ehr_forecast.submission import validate_predictions, write_predictions


def test_rare_targets_have_explicit_metric_policy():
    labels = np.array([[0, 0, 1], [1, 0, 1]])
    scores = np.array([[0.1, 0.2, 0.8], [0.9, 0.4, 0.9]])
    metrics = evaluate_predictions(labels, scores, ["01", "02", "03"])
    assert metrics["macro_auroc"] == 1.0
    assert metrics["n_auroc_valid"] == 1
    assert metrics["mAP"] == pytest.approx(2 / 3)
    assert metrics["map_positive_classes"] == 1.0
    assert metrics["n_ap_positive_classes"] == 2
    assert metrics["per_condition"][1]["auroc"] is None
    assert metrics["per_condition"][1]["average_precision"] == 0.0
    assert not metrics["per_condition"][1]["ap_has_positive_examples"]
    assert metrics["per_condition"][2]["auroc"] is None
    json.dumps(metrics, allow_nan=False)


def test_no_positive_targets_returns_defined_zero_ap():
    metrics = evaluate_predictions(np.zeros((2, 2)), np.full((2, 2), 0.5), ["a", "b"])
    assert metrics["macro_auroc"] is None
    assert metrics["micro_auroc"] is None
    assert metrics["mAP"] == metrics["micro_ap"] == 0.0
    assert metrics["map_positive_classes"] is None
    assert metrics["brier"] == 0.25
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("scores", [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1.01]]), np.array([[-0.01]])])
def test_metric_rejects_invalid_probabilities(scores):
    with pytest.raises(ValueError):
        evaluate_predictions([[1]], scores, ["a"])


@pytest.mark.parametrize("labels", [[[0.3]], [[np.nan]], [], [[1, 0]]])
def test_metric_rejects_invalid_labels(labels):
    with pytest.raises(ValueError):
        evaluate_predictions(labels, [[0.5]], ["a"])


def test_score_ties_use_average_precision_ranking():
    metrics = evaluate_predictions([[0], [1], [0], [1]], [[0.5]] * 4, ["a"])
    assert metrics["macro_auroc"] == 0.5
    assert metrics["mAP"] == 0.5


def test_csv_round_trip_preserves_codes_and_exact_coverage(tmp_path):
    path = tmp_path / "predictions.csv"
    ids = np.array(["patient_b", "patient_a"])
    result = write_predictions(path, ids, [[0.2, 0.8], [0.4, 0.6]], ["001", "999"], ["patient_a", "patient_b"])
    assert result["valid"]
    assert result["n_patients"] == 2
    assert result["n_targets"] == 2
    assert path.read_text().splitlines()[0] == "patient_id,001,999"
    assert validate_predictions(path, ["001", "999"], ids)["sha256"] == result["sha256"]


@pytest.mark.parametrize(
    "contents,error",
    [
        ("patient_id,a,a\np,0.2,0.4\n", "duplicate column"),
        ("patient_id,b,a\np,0.2,0.4\n", "CSV columns"),
        ("patient_id,a,b\np,0.2,0.4\np,0.3,0.6\n", "duplicate patient"),
        ("patient_id,a,b\nother,0.2,0.4\n", "coverage mismatch"),
        ("patient_id,a,b\np,nan,0.4\n", "NaN"),
        ("patient_id,a,b\np,0.2,1.1\n", "between 0 and 1"),
        ("patient_id,a,b\np,0.2\n", "fields"),
        ("patient_id,a,b\np,no,0.4\n", "Non-numeric"),
        ("patient_id,a,b\n", "cannot be empty"),
    ],
)
def test_csv_validator_rejects_corrupt_submission(tmp_path, contents, error):
    path = tmp_path / "bad.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        validate_predictions(path, ["a", "b"], ["p"])


def test_failed_write_does_not_replace_existing_file(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("existing artifact")
    with pytest.raises(ValueError, match="coverage mismatch"):
        write_predictions(path, ["p"], [[0.2]], ["a"], ["q"])
    assert path.read_text() == "existing artifact"
