import copy
import json

import numpy as np
import pytest

from ehr_forecast.factorial_audit import CELLS, COEFFICIENTS, factorial_analysis
from ehr_forecast.metrics import evaluate_predictions


SEEDS = [17, 42, 73]


@pytest.fixture
def additive():
    y = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1]])
    bad = .9 - .8 * y
    good = .1 + .8 * y
    cells = {cell: {} for cell in CELLS}
    for seed in SEEDS:
        cells["C0R0"][seed] = bad.copy()
        cells["C1R0"][seed] = np.column_stack([good[:, 0], bad[:, 1]])
        cells["C0R1"][seed] = np.column_stack([bad[:, 0], good[:, 1]])
        cells["C1R1"][seed] = good.copy()
    return y, cells, ["001", "002"]


def test_additive_cells_have_zero_interaction_and_separate_main_effects(additive):
    y, cells, codes = additive
    result = factorial_analysis(y, cells, codes, n_bootstrap=16)
    contrasts = result["overall"]["contrasts"]
    bad = evaluate_predictions(y, cells["C0R0"][17], codes)
    good = evaluate_predictions(y, cells["C1R1"][17], codes)
    for metric in ("mAP", "macro_auroc", "brier"):
        assert contrasts["C_main"]["mean_single_model_contrast"][metric] == pytest.approx((good[metric] - bad[metric]) / 2)
        assert contrasts["R_main"]["mean_single_model_contrast"][metric] == pytest.approx((good[metric] - bad[metric]) / 2)
        assert contrasts["interaction"]["mean_single_model_contrast"][metric] == pytest.approx(0, abs=1e-14)
        assert contrasts["interaction"]["intervals"][metric]["ci_lower"] == pytest.approx(0, abs=1e-14)
        assert contrasts["interaction"]["intervals"][metric]["ci_upper"] == pytest.approx(0, abs=1e-14)
    assert contrasts["C_main"]["wins"]["mAP"]["wins"] == 3
    assert contrasts["C_main"]["wins"]["brier"]["wins"] == 3
    assert result["uses_probability_ensemble"] is False
    assert result["selection"] is None
    json.dumps(result, allow_nan=False)


def test_interaction_is_difference_in_differences(additive):
    y, cells, codes = additive
    for cell in ("C1R0", "C0R1"):
        cells[cell] = copy.deepcopy(cells["C0R0"])
    result = factorial_analysis(y, cells, codes, n_bootstrap=16)["overall"]["contrasts"]
    change = evaluate_predictions(y, cells["C1R1"][17], codes)["mAP"] - evaluate_predictions(y, cells["C0R0"][17], codes)["mAP"]
    assert result["interaction"]["mean_single_model_contrast"]["mAP"] == pytest.approx(change)
    assert result["C_main"]["mean_single_model_contrast"]["mAP"] == pytest.approx(change / 2)
    assert result["R_main"]["mean_single_model_contrast"]["mAP"] == pytest.approx(change / 2)


def test_shared_bootstrap_matches_explicit_repeated_patients_with_ties():
    rng = np.random.default_rng(81)
    y = rng.integers(0, 2, size=(7, 3))
    y[:, 0] = 0
    codes = ["a", "b", "c"]
    cells = {cell: {seed: rng.integers(0, 3, size=y.shape) / 2 for seed in SEEDS} for cell in CELLS}
    actual = factorial_analysis(y, cells, codes, n_bootstrap=14, seed=24)["overall"]
    draw_rng = np.random.default_rng(24)
    expected = {name: {metric: [] for metric in ("mAP", "macro_auroc", "brier")} for name in COEFFICIENTS}
    for _ in range(14):
        indices = draw_rng.integers(0, len(y), size=len(y))
        points = {cell: {seed: evaluate_predictions(y[indices], cells[cell][seed][indices], codes) for seed in SEEDS} for cell in CELLS}
        for name, weights in COEFFICIENTS.items():
            for metric in expected[name]:
                if points["C0R0"][17][metric] is None:
                    continue
                values = [sum(weights[cell] * points[cell][seed][metric] for cell in CELLS) for seed in SEEDS]
                expected[name][metric].append(np.mean(values))
    for name, metrics in expected.items():
        for metric, values in metrics.items():
            interval = actual["contrasts"][name]["intervals"][metric]
            np.testing.assert_allclose([interval["ci_lower"], interval["ci_upper"]], np.quantile(values, [.025, .975]), atol=1e-13)
            assert interval["bootstrap_mean"] == pytest.approx(np.mean(values), abs=1e-13)


def test_contrasts_are_mean_single_model_metrics_not_probability_ensemble():
    y = np.array([[0], [1], [0], [1]])
    baseline = {17: np.array([[.1], [.4], [.3], [.2]]), 42: np.array([[.9], [.2], [.1], [.8]]),
                73: np.array([[.4], [.3], [.2], [.1]])}
    improved = {seed: y * .8 + .1 for seed in SEEDS}
    cells = {"C0R0": baseline, "C1R0": improved, "C0R1": baseline, "C1R1": improved}
    result = factorial_analysis(y, cells, ["a"], n_bootstrap=12)["overall"]
    expected = np.mean([evaluate_predictions(y, improved[seed], ["a"])["mAP"] - evaluate_predictions(y, baseline[seed], ["a"])["mAP"] for seed in SEEDS])
    wrong = evaluate_predictions(y, np.mean(list(improved.values()), axis=0), ["a"])["mAP"] - evaluate_predictions(y, np.mean(list(baseline.values()), axis=0), ["a"])["mAP"]
    assert result["contrasts"]["C_main"]["mean_single_model_contrast"]["mAP"] == pytest.approx(expected)
    assert expected != pytest.approx(wrong)


def test_seed_order_and_per_condition_contributions_are_stable(additive):
    y, cells, codes = additive
    first = factorial_analysis(y, cells, codes, n_bootstrap=10)
    reordered = {cell: dict(reversed(list(cells[cell].items()))) for cell in reversed(CELLS)}
    assert factorial_analysis(y, reordered, codes, n_bootstrap=10) == first
    for contrast in first["overall"]["contrasts"].values():
        assert sum(row["contribution_to_mAP_contrast"] for row in contrast["per_condition_ap_contributions"]) == pytest.approx(contrast["mean_single_model_contrast"]["mAP"])


def test_strict_metadata_rejects_patient_or_code_order(additive):
    y, cells, codes = additive
    ids = [f"p{index}" for index in range(len(y))]
    metadata = {cell: {seed: {"probabilities": scores, "patient_ids": ids.copy(), "target_codes": codes.copy()}
                       for seed, scores in runs.items()} for cell, runs in cells.items()}
    result = factorial_analysis(y, metadata, codes, patient_ids=ids, require_alignment_metadata=True, n_bootstrap=8)
    assert result["alignment"]["strict_all_predictions"]
    wrong = copy.deepcopy(metadata)
    wrong["C1R1"][42]["patient_ids"].reverse()
    with pytest.raises(ValueError, match="patient order"):
        factorial_analysis(y, wrong, codes, patient_ids=ids, n_bootstrap=8)
    wrong = copy.deepcopy(metadata)
    wrong["C0R1"][17]["target_codes"].reverse()
    with pytest.raises(ValueError, match="CODE order"):
        factorial_analysis(y, wrong, codes, patient_ids=ids, n_bootstrap=8)
    with pytest.raises(ValueError, match="explicit patient/CODE"):
        factorial_analysis(y, cells, codes, patient_ids=ids, require_alignment_metadata=True, n_bootstrap=8)
    with pytest.raises(ValueError, match="reference patient_ids"):
        factorial_analysis(y, metadata, codes, require_alignment_metadata=True, n_bootstrap=8)
    inferred = factorial_analysis(y, metadata, codes, n_bootstrap=8)
    assert inferred["alignment"]["all_prediction_metadata_present"]
    assert not inferred["alignment"]["strict_all_predictions"]
    assert not inferred["alignment"]["label_order_reference_provided"]


def test_fixed_groups_and_at_risk_bce_match_direct_cohort(additive):
    y, cells, codes = additive
    ids = [f"p{index}" for index in range(len(y))]
    prior = np.zeros_like(y, dtype=bool)
    prior[0] = True
    mask = np.array([True, True, True, False, False, False])
    grouped = factorial_analysis(y, cells, codes, groups={"long": ids[:3], "short": ~mask, "empty": []},
                                patient_ids=ids, prior_conditions=prior, n_bootstrap=12)
    direct = factorial_analysis(y[mask], {cell: {seed: scores[mask] for seed, scores in runs.items()} for cell, runs in cells.items()},
                               codes, prior_conditions=prior[mask], n_bootstrap=12)
    assert grouped["groups"]["long"] == direct["overall"]
    assert grouped["group_patient_counts"] == {"long": 3, "short": 3, "empty": 0}
    assert grouped["groups"]["empty"] is None
    own = grouped["overall"]["cells"]["C0R0"]["per_seed"][0]["metrics"]["at_risk_bce"]
    probabilities = cells["C0R0"][17]
    losses = -(y * np.log(probabilities) + (1-y) * np.log1p(-probabilities))
    assert own == pytest.approx(losses[~prior].mean())


def test_no_positive_or_at_risk_examples_use_explicit_undefined_metrics():
    y = np.zeros((5, 2))
    cells = {cell: {seed: np.full_like(y, .1 if cell == "C0R0" else .8) for seed in SEEDS} for cell in CELLS}
    result = factorial_analysis(y, cells, ["a", "b"], prior_conditions=np.ones_like(y), n_bootstrap=10)["overall"]
    for contrast in result["contrasts"].values():
        assert contrast["mean_single_model_contrast"]["mAP"] == 0
        for metric in ("macro_auroc", "at_risk_bce"):
            assert contrast["intervals"][metric]["estimate"] is None
            assert contrast["intervals"][metric]["n_resamples_defined"] == 0
    assert result["bootstrap_n_auroc_valid"]["max"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("control", [
    {"n_bootstrap": 1}, {"n_bootstrap": True}, {"seed": -1}, {"confidence": 1},
    {"expected_seeds": [17, 42]}, {"expected_seeds": [17, 42, 73, 73]},
    {"groups": [1, 2]}, {"patient_ids": ["p", "p"]},
])
def test_invalid_controls_are_rejected(additive, control):
    y, cells, codes = additive
    with pytest.raises(ValueError):
        factorial_analysis(y, cells, codes, **control)


def test_missing_cells_seeds_and_invalid_scores_are_rejected(additive):
    y, cells, codes = additive
    with pytest.raises(ValueError, match="four factorial"):
        factorial_analysis(y, {key: value for key, value in cells.items() if key != "C1R1"}, codes)
    wrong = copy.deepcopy(cells)
    wrong["C1R1"].pop(73)
    with pytest.raises(ValueError, match="complete seed"):
        factorial_analysis(y, wrong, codes)
    wrong = copy.deepcopy(cells)
    wrong["C1R1"][17][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        factorial_analysis(y, wrong, codes)
    wrong = copy.deepcopy(cells)
    wrong["C1R1"][17] = np.zeros((len(y)-1, len(codes)))
    with pytest.raises(ValueError, match="probability shape"):
        factorial_analysis(y, wrong, codes)


def test_invalid_prior_and_group_ids_rejected(additive):
    y, cells, codes = additive
    with pytest.raises(ValueError, match="overlap prior"):
        factorial_analysis(y, cells, codes, prior_conditions=np.ones_like(y))
    with pytest.raises(ValueError, match="binary matrix"):
        factorial_analysis(y, cells, codes, prior_conditions=np.full_like(y, 2))
    ids = [f"p{index}" for index in range(len(y))]
    with pytest.raises(ValueError, match="unknown validation"):
        factorial_analysis(y, cells, codes, groups={"long": ["outside"]}, patient_ids=ids, n_bootstrap=2)
