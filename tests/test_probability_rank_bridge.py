import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score


spec = importlib.util.spec_from_file_location("probability_rank_bridge", Path(__file__).parents[1] / "scripts/analyze_probability_rank_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def effects(a, b, y):
    ab, ba = bridge.transplant(a, b)
    values = [bridge.losses(y, p)["Brier"].mean() for p in (a, b, ab, ba)]
    return bridge.symmetric_effects(*values), ab, ba


def test_monotonic_score_change_changes_loss_without_changing_ap_or_order_effect():
    a = np.array([.05, .2, .4, .8]); b = a**2; y = np.array([0, 1, 0, 1])
    e, ab, ba = effects(a, b, y)
    assert e["patient_order_effect"] == pytest.approx(0, abs=1e-15)
    assert average_precision_score(y, a) == average_precision_score(y, b)
    np.testing.assert_array_equal(ab, b); np.testing.assert_array_equal(ba, a)


def test_permuted_multiset_has_zero_distribution_effect_and_preserves_cross_ap():
    a = np.array([.05, .2, .4, .8]); b = a[[3, 2, 0, 1]]; y = np.array([0, 1, 0, 1])
    e, ab, ba = effects(a, b, y)
    assert e["score_distribution_effect"] == pytest.approx(0, abs=1e-15)
    assert e["patient_order_effect"] == pytest.approx(e["total_delta"])
    assert average_precision_score(y, ab) == average_precision_score(y, a)
    assert average_precision_score(y, ba) == average_precision_score(y, b)


def test_mixed_changes_reconstruct_exact_total():
    a = np.array([.05, .2, .4, .8]); b = np.array([.3, .9, .1, .6]); y = np.array([0, 1, 0, 1])
    e, _, _ = effects(a, b, y)
    assert e["score_distribution_effect"] + e["patient_order_effect"] == pytest.approx(e["total_delta"], abs=1e-15)


def test_ties_are_refused_rather_than_broken_by_patient_order():
    with pytest.raises(ValueError, match="Ties"):
        bridge.transplant([.1, .1, .3], [.1, .2, .4])


def test_brier_uses_raw_probabilities_and_bce_remains_finite_at_boundaries():
    value = bridge.losses(np.array([0, 1, 1]), np.array([0., 1., 0.]))
    np.testing.assert_array_equal(value["Brier"], [0., 0., 1.])
    assert np.isfinite(value["BCE"]).all()
