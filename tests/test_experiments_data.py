"""Length changes must not silently refit an encoder or incident labels."""

import csv
from datetime import date, timedelta
import json

import numpy as np
import pytest

from ehr_forecast.experiments_data import analyze_histories, prepare_length_ablation, prepare_summary_dataset
from test_data import fixture_data


def test_length_ablation_has_common_encoder_and_identical_patient_targets(tmp_path):
    source = fixture_data(tmp_path / "source")
    result = prepare_length_ablation(source, tmp_path / "out", lengths=(2, 4, 8), min_token_count=1)
    assert result["same_encoder_for_all_lengths"]
    directories = [tmp_path / "out" / f"prepared_l{length}" for length in (2, 4, 8)]
    for name in ("vocabulary.json", "preprocessing.json"):
        assert len({(directory / name).read_bytes() for directory in directories}) == 1
    vocab = json.loads((directories[0] / "vocabulary.json").read_text())["token_to_id"]
    # Even the shortest sequence may use the older training concept in its
    # common encoder, because fitting scope is independent of retained length.
    assert "COND:old" in vocab
    assert not any("future" in token or "COND:new" in token for token in vocab)
    recipe = json.loads((directories[0] / "preprocessing.json").read_text())
    assert recipe["numeric_bin_edges"]["OBS:lab|UNIT=mg"] == [10.0]
    for split in ("train", "val", "test"):
        variants = [np.load(directory / f"{split}.npz", allow_pickle=False) for directory in directories]
        for key in ("patient_ids", "static", "prior_conditions", "anchor_dates", "full_history_length"):
            for variant in variants[1:]:
                np.testing.assert_array_equal(variants[0][key], variant[key])
        if split != "test":
            for variant in variants[1:]:
                np.testing.assert_array_equal(variants[0]["labels"], variant["labels"])
        else:
            assert all("labels" not in variant.files for variant in variants)
        full_tokens = variants[-1]["tokens"][0, 1:]
        full_tokens = full_tokens[full_tokens != 0]
        for length, variant in zip((2, 4, 8), variants):
            np.testing.assert_array_equal(variant["time_days"][:, 0], 0)
            retained = variant["tokens"][0, 1:]
            np.testing.assert_array_equal(retained[retained != 0], full_tokens[-(length - 1):])


def test_changing_length_set_does_not_change_fitted_recipe(tmp_path):
    source = fixture_data(tmp_path / "source")
    prepare_length_ablation(source, tmp_path / "first", lengths=(2, 4), min_token_count=1)
    prepare_length_ablation(source, tmp_path / "second", lengths=(4, 8), min_token_count=1)
    for name in ("vocabulary.json", "preprocessing.json"):
        assert (tmp_path / "first" / "prepared_l4" / name).read_bytes() == (tmp_path / "second" / "prepared_l4" / name).read_bytes()


def test_future_or_validation_changes_cannot_fit_ablation_encoder(tmp_path):
    source = fixture_data(tmp_path / "source")
    prepare_length_ablation(source, tmp_path / "before", lengths=(2, 4), min_token_count=1)
    with (source / "train_val" / "observations.csv").open("a", encoding="utf-8") as handle:
        handle.write("b,2012-01-01,val_unique,numeric,999999999,mg\n")
        handle.write("a,2019-01-01,future_unique,numeric,999999999,mg\n")
    prepare_length_ablation(source, tmp_path / "after", lengths=(2, 4), min_token_count=1)
    for name in ("vocabulary.json", "preprocessing.json"):
        assert (tmp_path / "before" / "prepared_l2" / name).read_bytes() == (tmp_path / "after" / "prepared_l2" / name).read_bytes()


def test_diagnostics_explains_loss_without_hidden_test_labels(tmp_path):
    source = fixture_data(tmp_path / "source")
    result = analyze_histories(source, tmp_path / "out", lengths=(2, 4, 256))
    assert result["splits"]["train"]["full_event_counts"]["max"] == 3
    assert result["splits"]["train"]["lengths"]["2"]["dropped_events"] == 2
    assert result["splits"]["train"]["lengths"]["2"]["patients_losing_unique_conditions"] == 1
    assert result["splits"]["test"]["patients"] == 1
    with (tmp_path / "out" / "history_patient_groups.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["patient_id"] for row in rows] == ["a", "b", "c"]
    assert {row["history_group_256"] for row in rows} == {"short"}


def test_ablation_protects_existing_cache_and_validates_lengths(tmp_path):
    source = fixture_data(tmp_path / "source")
    with pytest.raises(ValueError, match="lengths"):
        prepare_length_ablation(source, tmp_path / "out", lengths=(1,))
    prepare_length_ablation(source, tmp_path / "out", lengths=(2,))
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_length_ablation(source, tmp_path / "out", lengths=(2,))


def test_full_history_summary_survives_sequence_truncation_and_documents_features(tmp_path):
    source = fixture_data(tmp_path / "source")
    metadata = prepare_summary_dataset(source, tmp_path / "summary", max_events=2, min_token_count=1)
    assert metadata["full_history_summary"]
    assert metadata["n_static"] == 5 + metadata["summary_feature_count"]
    assert len(metadata["static_feature_names"]) == metadata["n_static"]
    train = np.load(tmp_path / "summary" / "train.npz", allow_pickle=False)
    test = np.load(tmp_path / "summary" / "test.npz", allow_pickle=False)
    np.testing.assert_array_equal(train["labels"], [[0, 1]])
    assert train["prior_conditions"][0, 0]
    assert "labels" not in test.files
    assert train["tokens"].shape == (1, 2)
    recipe = json.loads((tmp_path / "summary" / "preprocessing.json").read_text())["full_history_summary"]
    index = recipe["feature_names"].index("summary_standardized::log1p_all_count::COND:old")
    assert recipe["mean"][index] == pytest.approx(np.log(2))
    # Three events exist, but only the newest fits the sequence. The old
    # diagnosis count remains a full-history feature even after truncation.
    assert int(train["full_history_length"][0]) == 3
    assert recipe["fit_split"] == "train"


def test_summary_scaler_and_vocabulary_ignore_validation_and_future_values(tmp_path):
    source = fixture_data(tmp_path / "source")
    prepare_summary_dataset(source, tmp_path / "before", max_events=2, min_token_count=1)
    with (source / "train_val" / "observations.csv").open("a", encoding="utf-8") as handle:
        handle.write("b,2012-01-01,val_unique,numeric,999999999,mg\n")
        handle.write("a,2019-01-01,future_unique,numeric,999999999,mg\n")
    prepare_summary_dataset(source, tmp_path / "after", max_events=2, min_token_count=1)
    for filename in ("preprocessing.json", "vocabulary.json"):
        assert (tmp_path / "before" / filename).read_bytes() == (tmp_path / "after" / filename).read_bytes()
    before = np.load(tmp_path / "before" / "train.npz", allow_pickle=False)
    after = np.load(tmp_path / "after" / "train.npz", allow_pickle=False)
    np.testing.assert_array_equal(before["static"], after["static"])


def test_summary_clip_is_fixed_and_preserves_demographics_and_scaler(tmp_path):
    source = fixture_data(tmp_path / "source")
    # A validation-only abundance of a train-known token creates a large z-score.
    # Those extra patients/events must not alter the fitted training mean/std.
    with (source / "train_val" / "observations.csv").open("a", encoding="utf-8") as handle:
        for index in range(1000):
            day = date(2010, 1, 1) + timedelta(days=index)
            handle.write(f"b,{day.isoformat()},lab,numeric,100000,mg\n")
    plain_metadata = prepare_summary_dataset(source, tmp_path / "plain", max_events=2, min_token_count=1)
    clip_metadata = prepare_summary_dataset(source, tmp_path / "clip", max_events=2, min_token_count=1, standardized_clip=5)
    plain_recipe = json.loads((tmp_path / "plain" / "preprocessing.json").read_text())
    clip_recipe = json.loads((tmp_path / "clip" / "preprocessing.json").read_text())
    assert "summary_standardized_clip" not in plain_metadata
    assert "standardized_clip" not in plain_recipe["full_history_summary"]
    assert clip_metadata["summary_standardized_clip"] == 5
    assert clip_recipe["full_history_summary"].pop("standardized_clip") == 5
    assert plain_recipe == clip_recipe  # No per-validation fitting or alternate scaler.
    for split in ("train", "val", "test"):
        plain = np.load(tmp_path / "plain" / f"{split}.npz", allow_pickle=False)
        clipped = np.load(tmp_path / "clip" / f"{split}.npz", allow_pickle=False)
        np.testing.assert_array_equal(plain["static"][:, :5], clipped["static"][:, :5])
        np.testing.assert_array_equal(np.clip(plain["static"][:, 5:], -5, 5), clipped["static"][:, 5:])
        assert np.abs(clipped["static"][:, 5:]).max() <= 5
        for key in ("patient_ids", "tokens", "time_days", "prior_conditions"):
            np.testing.assert_array_equal(plain[key], clipped[key])
    val_plain = np.load(tmp_path / "plain" / "val.npz", allow_pickle=False)
    assert np.abs(val_plain["static"][:, 5:]).max() > 5


def test_clipped_summary_recipe_ignores_validation_and_future_values(tmp_path):
    source = fixture_data(tmp_path / "source")
    prepare_summary_dataset(source, tmp_path / "before", max_events=2, min_token_count=1, standardized_clip=5)
    with (source / "train_val" / "observations.csv").open("a", encoding="utf-8") as handle:
        handle.write("b,2012-01-01,val_unique,numeric,999999999,mg\n")
        handle.write("a,2019-01-01,future_unique,numeric,999999999,mg\n")
    prepare_summary_dataset(source, tmp_path / "after", max_events=2, min_token_count=1, standardized_clip=5)
    for filename in ("preprocessing.json", "vocabulary.json"):
        assert (tmp_path / "before" / filename).read_bytes() == (tmp_path / "after" / filename).read_bytes()
    before = np.load(tmp_path / "before" / "train.npz", allow_pickle=False)
    after = np.load(tmp_path / "after" / "train.npz", allow_pickle=False)
    np.testing.assert_array_equal(before["static"], after["static"])


@pytest.mark.parametrize("clip", [0, -1, float("nan"), float("inf")])
def test_summary_clip_rejects_nonpositive_or_nonfinite_values(tmp_path, clip):
    with pytest.raises(ValueError, match="standardized_clip"):
        prepare_summary_dataset(tmp_path / "source", tmp_path / "out", standardized_clip=clip)
