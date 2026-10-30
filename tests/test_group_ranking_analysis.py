"""Independent checks of analysis selection, patient pairing and gate rules."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_group_ranking.py"
SPEC = importlib.util.spec_from_file_location("group_ranking_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def rows(scores):
    return [{"epoch": i, "val/mAP": p} for i, p in enumerate(scores)]


def test_epoch0_fallback_is_an_actual_selection_candidate():
    selected = analysis.selection_from_history(rows([.5] + [.49]*15), patience=15, epochs=100)
    assert selected == {"best_epoch": 0, "best_validation_mAP": .5, "last_epoch": 15, "no_update_selected": True}


def test_selection_min_improvement_and_patience_are_replayed():
    selected = analysis.selection_from_history(rows([.5, .5000005, .6] + [.55]*3), patience=3, epochs=100)
    assert selected["best_epoch"] == 2 and selected["last_epoch"] == 5
    with pytest.raises(ValueError, match="stopped before"):
        analysis.selection_from_history(rows([.5,.6,.55]),patience=15,epochs=100)
    with pytest.raises(ValueError, match="contiguous"):
        analysis.selection_from_history([{"epoch":1,"val/mAP":.5}])


def toy_predictions():
    y = np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]],float)
    good = np.array([[.8,.2],[.1,.8],[.7,.7],[.2,.1],[.6,.3],[.3,.6]])
    bad = 1-good
    predictions = {phase:{m:{} for m in analysis.METHODS} for phase in ("best","epoch10")}
    for phase in predictions:
        for method in predictions[phase]:
            for seed in analysis.SEEDS:
                predictions[phase][method][seed] = good.copy() if seed==17 else bad.copy()
    return y, np.zeros_like(y,bool), predictions


def test_shared_patient_bootstrap_is_not_probability_ensembling():
    y, prior, predictions = toy_predictions()
    # Three fits have deliberately different quality. All methods match their
    # paired BCE fit, so every draw and point effect must be exactly zero.
    got = analysis.shared_bootstrap(y,prior,predictions,["A","B"],{"overall":np.ones(len(y),bool)},n_bootstrap=20)
    for phase in predictions:
        for method in analysis.METHODS[1:]:
            value=got["overall"]["comparisons"][phase][method]
            assert value["between_seed"]["mAP"]["mean"]==0
            for metric,ci in value["intervals"].items():
                assert ci["estimate"]==ci["ci_lower"]==ci["ci_upper"]==0, metric


def test_point_effect_averages_per_seed_metrics():
    y,prior,predictions=toy_predictions()
    predictions["best"]["ranknet"][42]=1-predictions["best"]["bce"][42]
    got=analysis.shared_bootstrap(y,prior,predictions,["A","B"],{"overall":np.ones(len(y),bool)},n_bootstrap=5)
    expected=np.mean([analysis.score(y,predictions["best"]["ranknet"][s],prior,["A","B"])["mAP"]-
                      analysis.score(y,predictions["best"]["bce"][s],prior,["A","B"])["mAP"] for s in analysis.SEEDS])
    value=got["overall"]["comparisons"]["best"]["ranknet"]
    assert value["between_seed"]["mAP"]["mean"]==expected
    assert value["per_seed_delta"][0]["mAP"]==0 and value["per_seed_delta"][2]["mAP"]==0


def test_skip_reproduction_cannot_be_used_for_formal_outputs(tmp_path):
    with pytest.raises(ValueError,match="restricted"):
        analysis.check_reproduction(tmp_path/"artifacts/group_ranking",{},tmp_path,skip=True,development=True)
    with pytest.raises(ValueError,match="restricted"):
        analysis.check_reproduction(tmp_path/"development",{},tmp_path,skip=True,development=False)
    assert analysis.check_reproduction(tmp_path/"development",{},tmp_path,skip=True,development=True)["skipped_for_development"]
