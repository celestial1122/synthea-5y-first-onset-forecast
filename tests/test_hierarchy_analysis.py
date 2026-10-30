import json

import numpy as np
import pytest

from ehr_forecast.hierarchy_analysis import paired_seed_bootstrap, paired_seed_group_comparisons
from ehr_forecast.metrics import evaluate_predictions


@pytest.fixture
def paired():
    labels = np.array([[0], [1], [0], [1]])
    a = {17: np.array([[.1], [.4], [.3], [.2]]),
         42: np.array([[.9], [.2], [.1], [.8]]),
         73: np.array([[.4], [.3], [.2], [.1]])}
    b = {model_seed: labels * .8 + .1 for model_seed in a}
    return labels, a, b, ["001"]


def test_estimand_is_mean_single_fit_difference_not_ensemble(paired):
    y, a, b, codes = paired
    result = paired_seed_bootstrap(y, a, b, codes, n_bootstrap=20, expected_seeds=[17, 42, 73])
    expected = np.mean([evaluate_predictions(y, b[s], codes)["mAP"] - evaluate_predictions(y, a[s], codes)["mAP"] for s in a])
    ensemble_delta = evaluate_predictions(y, np.mean(list(b.values()), axis=0), codes)["mAP"] - evaluate_predictions(y, np.mean(list(a.values()), axis=0), codes)["mAP"]
    assert result["delta_mAP"]["estimate"] == pytest.approx(expected)
    assert result["delta_mAP"]["estimate"] != pytest.approx(ensemble_delta)
    assert result["uses_probability_ensemble"] is False
    assert result["mean_paired_delta"]["mAP"] == pytest.approx(expected)
    assert result["wins"]["mAP"] == {"wins": 3, "ties": 0, "losses": 0, "n_defined": 3, "tie_tolerance": 1e-12}
    assert result["between_seed_deltas"]["mAP"]["sample_std"] == pytest.approx(np.std([row["delta_mAP"] for row in result["per_seed"]], ddof=1))
    json.dumps(result, allow_nan=False)


def test_shared_patient_bootstrap_matches_explicit_resamples_with_ties():
    rng = np.random.default_rng(81)
    y = rng.integers(0, 2, size=(9, 3))
    y[:, 0] = 0
    a = {s: rng.integers(0, 3, size=y.shape) / 2 for s in [17, 42, 73]}
    b = {s: rng.integers(0, 3, size=y.shape) / 2 for s in [17, 42, 73]}
    codes = ["a", "b", "c"]
    result = paired_seed_bootstrap(y, a, b, codes, n_bootstrap=20, seed=24)
    resampling = np.random.default_rng(24)
    map_deltas, auc_deltas = [], []
    for _ in range(20):
        indices = resampling.integers(0, len(y), size=len(y))
        metrics_a = [evaluate_predictions(y[indices], a[s][indices], codes) for s in sorted(a)]
        metrics_b = [evaluate_predictions(y[indices], b[s][indices], codes) for s in sorted(a)]
        map_deltas.append(np.mean([mb["mAP"] - ma["mAP"] for ma, mb in zip(metrics_a, metrics_b)]))
        if metrics_a[0]["macro_auroc"] is not None:
            auc_deltas.append(np.mean([mb["macro_auroc"] - ma["macro_auroc"] for ma, mb in zip(metrics_a, metrics_b)]))
    np.testing.assert_allclose([result["delta_mAP"]["ci_lower"], result["delta_mAP"]["ci_upper"]], np.quantile(map_deltas, [.025, .975]), atol=1e-14)
    np.testing.assert_allclose([result["delta_macro_auroc"]["ci_lower"], result["delta_macro_auroc"]["ci_upper"]], np.quantile(auc_deltas, [.025, .975]), atol=1e-14)
    assert result["delta_mAP"]["bootstrap_mean"] == pytest.approx(np.mean(map_deltas), abs=1e-14)


def test_seed_mapping_order_does_not_change_pairing(paired):
    y, a, b, codes = paired
    result = paired_seed_bootstrap(y, a, b, codes, n_bootstrap=20)
    reordered = paired_seed_bootstrap(y, dict(reversed(list(a.items()))), dict(reversed(list(b.items()))), codes, n_bootstrap=20)
    assert reordered == result
    assert result["seeds"] == [17, 42, 73]


def test_identical_families_have_zero_single_model_interval(paired):
    y, a, _, codes = paired
    result = paired_seed_bootstrap(y, a, a, codes, n_bootstrap=20)
    for metric in ("delta_mAP", "delta_macro_auroc"):
        assert result[metric]["estimate"] == 0
        assert result[metric]["ci_lower"] == result[metric]["ci_upper"] == 0


def test_fixed_group_mask_uses_only_selected_patients(paired):
    y, a, b, codes = paired
    mask = np.array([True, True, True, False])
    selected = paired_seed_bootstrap(y, a, b, codes, mask=mask, n_bootstrap=20)
    direct = paired_seed_bootstrap(y[mask], {s: v[mask] for s, v in a.items()}, {s: v[mask] for s, v in b.items()}, codes, n_bootstrap=20)
    assert selected["delta_mAP"] == direct["delta_mAP"]
    assert selected["n_patients"] == 3
    assert selected["n_reference_patients"] == 4
    assert selected["n_positive_by_target"] == [1]


def test_no_positive_cohort_keeps_ap_zero_and_auc_undefined():
    y = np.zeros((6, 3))
    a = {s: np.full_like(y, .2) for s in [17, 42, 73]}
    b = {s: np.full_like(y, .8) for s in [17, 42, 73]}
    result = paired_seed_bootstrap(y, a, b, ["a", "b", "c"], n_bootstrap=20)
    assert result["delta_mAP"]["estimate"] == result["delta_mAP"]["ci_lower"] == 0
    assert result["delta_macro_auroc"]["estimate"] is None
    assert result["delta_macro_auroc"]["n_resamples_defined"] == 0
    assert result["between_seed_deltas"]["macro_auroc"]["mean"] is None
    assert result["bootstrap_n_auroc_valid"]["max"] == 0
    assert result["wins"]["macro_auroc"]["n_defined"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("changes", [
    {"n_bootstrap": 1}, {"n_bootstrap": True}, {"seed": -1}, {"confidence": 1},
    {"mask": [1, 1, 0, 0]}, {"mask": np.zeros(4, dtype=bool)},
    {"expected_seeds": [17, 42]}, {"expected_seeds": [17, 42, 73, 73]},
])
def test_invalid_controls_are_rejected(paired, changes):
    y, a, b, codes = paired
    with pytest.raises(ValueError):
        paired_seed_bootstrap(y, a, b, codes, **changes)


def test_missing_seed_or_bad_score_shape_rejected(paired):
    y, a, b, codes = paired
    with pytest.raises(ValueError, match="same paired seeds"):
        paired_seed_bootstrap(y, a, {17: b[17]}, codes)
    with pytest.raises(ValueError, match="Expected probability shape"):
        paired_seed_bootstrap(y, a, {**b, 17: np.zeros((3, 1))}, codes)
    with pytest.raises(ValueError, match="Model seeds"):
        paired_seed_bootstrap(y, {"17": a[17]}, b, codes)


def test_group_wrapper_accepts_fixed_ids_masks_and_empty_group(paired):
    y, a, b, codes = paired
    ids = ["p0", "p1", "p2", "p3"]
    groups = {"short": [True, True, False, False], "long": ["p2", "p3"], "empty": []}
    result = paired_seed_group_comparisons(y, a, b, codes, groups, ids, n_bootstrap=20,
                                           expected_seeds=[17, 42, 73])
    assert result["group_patient_counts"] == {"short": 2, "long": 2, "empty": 0}
    assert result["groups"]["empty"] is None
    assert result["groups"]["long"]["mean_paired_delta"]["mAP"] == pytest.approx(
        paired_seed_bootstrap(y, a, b, codes, mask=np.array([False, False, True, True]), n_bootstrap=20)["delta_mAP"]["estimate"])
    assert result["uses_probability_ensemble"] is False
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="unknown validation"):
        paired_seed_group_comparisons(y, a, b, codes, {"bad": ["outside"]}, ids, n_bootstrap=2)
    with pytest.raises(ValueError, match="uniquely match"):
        paired_seed_group_comparisons(y, a, b, codes, {}, ids[:-1], n_bootstrap=2)
