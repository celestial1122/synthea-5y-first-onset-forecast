import json
from pathlib import Path

import numpy as np
import pytest

from ehr_forecast.experiment_analysis import (
    _score_orders, _weighted_macro_metrics, average_validation_predictions,
    evaluate_groups, load_aligned_predictions, paired_bootstrap, summarize_runs,
)
from ehr_forecast.metrics import evaluate_predictions
from ehr_forecast.submission import write_predictions


def test_weighted_tie_metrics_match_explicit_patient_resampling():
    rng = np.random.default_rng(24)
    labels = rng.integers(0, 2, size=(19, 6))
    labels[:, 0] = 0
    labels[:, 1] = 1
    scores = rng.integers(0, 4, size=labels.shape) / 3
    orders = _score_orders(scores)
    for _ in range(20):
        indices = rng.integers(0, len(labels), size=len(labels))
        weights = np.bincount(indices, minlength=len(labels)).astype(float)
        fast = _weighted_macro_metrics(labels, orders, weights)
        expected = evaluate_predictions(labels[indices], scores[indices], [str(i) for i in range(6)])
        assert fast["mAP"] == pytest.approx(expected["mAP"], abs=1e-14)
        assert fast["macro_auroc"] == pytest.approx(expected["macro_auroc"], abs=1e-14)
        assert fast["n_auroc_valid"] == expected["n_auroc_valid"]


def test_identical_models_have_exactly_zero_paired_interval():
    labels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    scores = np.array([[.1, .2], [.8, .3], [.4, .7], [.9, .8]])
    result = paired_bootstrap(labels, scores, scores, ["a", "b"], n_bootstrap=50)
    assert result["delta_mAP"]["estimate"] == 0
    assert result["delta_mAP"]["ci_lower"] == result["delta_mAP"]["ci_upper"] == 0
    assert result["delta_macro_auroc"]["ci_lower"] == result["delta_macro_auroc"]["ci_upper"] == 0
    assert result == paired_bootstrap(labels, scores, scores, ["a", "b"], n_bootstrap=50)
    json.dumps(result, allow_nan=False)


def test_improvement_direction_and_shared_patient_resampling():
    labels = np.tile([[0], [1]], (20, 1))
    result = paired_bootstrap(labels, 1-labels, labels, ["a"], n_bootstrap=50)
    assert result["difference_direction"] == "B minus A"
    assert result["delta_mAP"]["estimate"] == .5
    assert result["delta_mAP"]["ci_lower"] > 0
    assert result["delta_macro_auroc"]["estimate"] == 1


def test_all_zero_cohort_has_zero_ap_and_explicitly_undefined_auc():
    result = paired_bootstrap(np.zeros((5, 2)), np.full((5, 2), .2), np.full((5, 2), .8), ["a", "b"], n_bootstrap=20)
    assert result["point_estimate"]["a"]["mAP"] == 0
    assert result["delta_mAP"]["ci_lower"] == 0
    assert result["delta_macro_auroc"]["estimate"] is None
    assert result["delta_macro_auroc"]["n_resamples_defined"] == 0
    assert result["bootstrap_n_auroc_valid"]["max"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"n_bootstrap": 1}, {"n_bootstrap": True}, {"confidence": 1}, {"seed": -1}])
def test_bootstrap_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        paired_bootstrap([[0], [1]], [[.1], [.9]], [[.2], [.8]], ["a"], **kwargs)


def test_groups_preserve_all_target_ap_and_report_small_support():
    labels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    scores = labels * .8 + .1
    groups = {"short": [True, True, False, False], "long": ["c", "d"], "empty": []}
    result = evaluate_groups(labels, scores, ["a", "b"], groups, ["a", "b", "c", "d"])
    short = result["groups"]["short"]
    assert short["n_patients"] == 2
    assert short["n_positive_by_target"] == [1, 0]
    assert short["metrics"]["mAP"] == .5
    assert short["metrics"]["n_auroc_valid"] == 1
    assert result["groups"]["empty"]["metrics"] is None
    with pytest.raises(ValueError, match="unknown"):
        evaluate_groups(labels, scores, ["a", "b"], {"bad": ["z"]}, ["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="one entry"):
        evaluate_groups(labels, scores, ["a", "b"], {"bad": [True]})


@pytest.fixture
def run_family(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    ids = ["patient_a", "patient_b", "patient_c", "patient_d"]
    labels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    metadata = {"target_codes": ["001", "999"], "target_descriptions": ["A", "B"]}
    (prepared / "metadata.json").write_text(json.dumps(metadata))
    np.savez(prepared / "val.npz", patient_ids=np.array(ids), labels=labels)
    scores = [np.array([[.2, .1], [.7, .3], [.4, .9], [.8, .6]]),
              np.array([[.1, .2], [.3, .4], [.6, .7], [.9, .8]])]
    directories = []
    for seed, probabilities in zip([42, 43], scores):
        directory = tmp_path / f"run{seed}"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"seed": seed}))
        (directory / "metadata.json").write_text(json.dumps(metadata))
        # Intentionally reverse each CSV's rows; analysis must use official order.
        write_predictions(directory / "val_predictions.csv", ids[::-1], probabilities[::-1], metadata["target_codes"])
        # This file must never be read by validation analysis.
        (directory / "test_predictions.csv").write_text("not valid test data")
        directories.append(directory)
    return prepared, directories, ids, labels, scores


def test_multiseed_summary_aligns_rows_and_uses_sample_std_all_seed_ensemble(run_family):
    prepared, directories, ids, labels, scores = run_family
    result = summarize_runs(directories[::-1], prepared, {"long": ids[2:]}, expected_seeds=[42, 43], family_name="fixed")
    assert result["seeds"] == [42, 43]
    expected = [evaluate_predictions(labels, score, ["001", "999"]) for score in scores]
    assert result["across_seeds"]["mAP"]["sample_std"] == pytest.approx(np.std([m["mAP"] for m in expected], ddof=1))
    assert result["across_seeds"]["mAP"]["mean"] == pytest.approx(np.mean([m["mAP"] for m in expected]))
    ensemble = (scores[0] + scores[1]) / 2
    assert result["equal_weight_ensemble"]["metrics"]["mAP"] == pytest.approx(evaluate_predictions(labels, ensemble, ["001", "999"])["mAP"])
    np.testing.assert_allclose(average_validation_predictions(directories, prepared), ensemble)
    np.testing.assert_allclose(load_aligned_predictions(directories[0] / "val_predictions.csv", ids, ["001", "999"]), scores[0])
    json.dumps(result, allow_nan=False)


def test_multiseed_summary_rejects_incomplete_duplicate_and_inconsistent_families(run_family):
    prepared, directories, _, _, _ = run_family
    with pytest.raises(ValueError, match="expected seeds"):
        summarize_runs(directories[:1], prepared, expected_seeds=[42, 43])
    config_path = directories[1] / "config.json"
    config_path.write_text(json.dumps({"seed": 42}))
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize_runs(directories, prepared)
    config_path.write_text(json.dumps({"seed": 43}))
    (directories[1] / "metadata.json").write_text(json.dumps({"target_codes": ["999", "001"]}))
    with pytest.raises(ValueError, match="target ordering"):
        summarize_runs(directories, prepared)


def test_one_seed_does_not_fabricate_zero_stability_variance(run_family):
    prepared, directories, _, _, _ = run_family
    result = summarize_runs(directories[:1], prepared)
    assert result["n_seeds"] == 1
    assert result["across_seeds"]["mAP"]["sample_std"] is None


def test_family_summary_does_not_mislabel_architecture_changes_as_seed_variance(run_family):
    prepared, directories, _, _, _ = run_family
    (directories[1] / "config.json").write_text(json.dumps({"seed": 43, "model_type": "different"}))
    with pytest.raises(ValueError, match="experimental family"):
        summarize_runs(directories, prepared)
