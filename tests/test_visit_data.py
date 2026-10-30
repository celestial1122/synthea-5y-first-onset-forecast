import json

import numpy as np
import pytest

from ehr_forecast.experiments_data import prepare_length_ablation
from ehr_forecast.visit_data import analyze_date_units, prepare_visit_datasets
from test_data import fixture_data, _csv


def _source_and_encoder(tmp_path):
    source = fixture_data(tmp_path / "source")
    prepare_length_ablation(source, tmp_path / "encoder", lengths=(8,), min_token_count=1)
    return source, tmp_path / "encoder/prepared_l8"


def test_date_compression_keeps_same_events_and_frozen_patient_information(tmp_path):
    source, encoder = _source_and_encoder(tmp_path)
    result = prepare_visit_datasets(source, encoder, tmp_path / "units", 3, 8, 3)
    assert result["same_vocabulary"]
    root = tmp_path / "units"
    paths = [root / name for name in ("prepared_raw", "prepared_date_recent", "prepared_date_full")]
    for path in paths:
        assert (path / "vocabulary.json").read_bytes() == (encoder / "vocabulary.json").read_bytes()
    for split in ("train", "val", "test"):
        with np.load(encoder / f"{split}.npz") as original:
            with np.load(paths[0] / f"{split}.npz") as raw, np.load(paths[1] / f"{split}.npz") as compressed:
                assert raw["tokens"].dtype == np.int16
                assert raw["tokens"].shape == (1, 3, 1)
                assert compressed["tokens"].shape == (1, 3, 8)
                np.testing.assert_array_equal(raw["tokens"][raw["tokens"] != 0], compressed["tokens"][compressed["tokens"] != 0])
            for path in paths:
                with np.load(path / f"{split}.npz") as arrays:
                    for key in ("patient_ids", "static", "prior_conditions", "anchor_dates", "full_history_length"):
                        np.testing.assert_array_equal(arrays[key], original[key])
                    if split != "test":
                        np.testing.assert_array_equal(arrays["labels"], original["labels"])
                    else:
                        assert "labels" not in arrays.files
                    assert (arrays["time_days"][arrays["tokens"].any(axis=2)] > 0).all()


def test_full_dates_recover_older_events_while_recent_control_has_exact_budget(tmp_path):
    source, encoder = _source_and_encoder(tmp_path)
    prepare_visit_datasets(source, encoder, tmp_path / "units", 2, 8, 2)
    with np.load(tmp_path / "units/prepared_raw/train.npz") as raw, np.load(tmp_path / "units/prepared_date_recent/train.npz") as recent, np.load(tmp_path / "units/prepared_date_full/train.npz") as full:
        assert raw["retained_event_count"].tolist() == [2]
        assert recent["retained_event_count"].tolist() == [2]
        assert full["retained_event_count"].tolist() == [3]
        assert full["retained_unit_count"].tolist() == [2]
        np.testing.assert_array_equal(raw["labels"], [[0, 1]])
        assert raw["prior_conditions"][0, 0]


def test_frozen_encoder_cannot_fit_validation_or_future_changes(tmp_path):
    source, encoder = _source_and_encoder(tmp_path)
    prepare_visit_datasets(source, encoder, tmp_path / "before", 4, 8, 3)
    with (source / "train_val/observations.csv").open("a", encoding="utf-8") as handle:
        handle.write("b,2012-01-01,validation_only,numeric,999999999,mg\n")
        handle.write("a,2019-01-01,future_only,numeric,999999999,mg\n")
    prepare_visit_datasets(source, encoder, tmp_path / "after", 4, 8, 3)
    for name in ("prepared_raw", "prepared_date_recent", "prepared_date_full"):
        for artifact in ("vocabulary.json", "preprocessing.json"):
            assert (tmp_path / "before" / name / artifact).read_bytes() == (tmp_path / "after" / name / artifact).read_bytes()
        with np.load(tmp_path / "before" / name / "train.npz") as before, np.load(tmp_path / "after" / name / "train.npz") as after:
            np.testing.assert_array_equal(before["tokens"], after["tokens"])
            np.testing.assert_array_equal(before["static"], after["static"])


def test_no_silent_within_date_truncation_and_no_overwrite(tmp_path):
    source, encoder = _source_and_encoder(tmp_path)
    with pytest.raises(ValueError, match="exceeding max_unit_events"):
        prepare_visit_datasets(source, encoder, tmp_path / "too_narrow", 3, 1, 3)
    prepare_visit_datasets(source, encoder, tmp_path / "good", 3, 8, 3)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_visit_datasets(source, encoder, tmp_path / "good", 3, 8, 3)


def test_empty_history_has_single_sentinel_without_fabricated_date(tmp_path):
    source = fixture_data(tmp_path / "source")
    _csv(source / "test/conditions.csv", ["PATIENT", "START", "CODE"], [])
    _csv(source / "test/observations.csv", ["PATIENT", "DATE", "CODE", "TYPE", "VALUE", "UNITS"], [])
    _csv(source / "test/encounters.csv", ["PATIENT", "START", "STOP", "CODE"], [])
    prepare_length_ablation(source, tmp_path / "encoder", lengths=(8,), min_token_count=1)
    prepare_visit_datasets(source, tmp_path / "encoder/prepared_l8", tmp_path / "units", 3, 8, 3)
    for name in ("prepared_raw", "prepared_date_recent", "prepared_date_full"):
        with np.load(tmp_path / "units" / name / "test.npz") as arrays:
            assert arrays["tokens"][0, 0, 0] == 2
            assert int(np.count_nonzero(arrays["tokens"])) == 1
            assert not np.any(arrays["time_days"])
            assert arrays["retained_event_count"].tolist() == [0]


def test_distinct_values_in_same_numeric_bin_keep_event_multiplicity(tmp_path):
    source, encoder = _source_and_encoder(tmp_path)
    with (source / "train_val/observations.csv").open("a", encoding="utf-8") as handle:
        handle.write("b,2011-01-01,lab,numeric,1000000,mg\n")
    prepare_visit_datasets(source, encoder, tmp_path / "units", 4, 8, 4)
    with np.load(tmp_path / "units/prepared_date_full/val.npz") as arrays:
        # The second day contains two distinct raw observations, even though
        # both fall above the same frozen training quantile edge.
        same_day = arrays["tokens"][0, 1]
        nonzero = same_day[same_day != 0]
        assert len(nonzero) == 2
        assert nonzero[0] == nonzero[1]


def test_date_diagnostics_and_parameter_validation(tmp_path):
    source, encoder = _source_and_encoder(tmp_path)
    stats = analyze_date_units(source, tmp_path / "diagnostics", unit_limits=(1, 2))
    assert stats["train"]["date_units"]["max"] == 2
    assert stats["train"]["daily_event_count"]["max"] == 2
    assert stats["train"]["lengths"]["1"]["event_retention"] == pytest.approx(1/3)
    assert stats["train"]["lengths"]["2"]["event_retention"] == 1
    with pytest.raises(ValueError, match="raw_event_limit"):
        prepare_visit_datasets(source, encoder, tmp_path / "bad", 2, 8, 3)
    vocabulary = json.loads((encoder / "vocabulary.json").read_text())
    vocabulary["token_to_id"]["too_large"] = 32768
    (encoder / "vocabulary.json").write_text(json.dumps(vocabulary))
    with pytest.raises(ValueError, match="int16"):
        prepare_visit_datasets(source, encoder, tmp_path / "bad_vocab", 3, 8, 3)
