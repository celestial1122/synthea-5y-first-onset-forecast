"""Meaningful temporal-leakage and train-only preprocessing regression cases."""

import csv
import json
from datetime import date

import numpy as np
import pytest

from ehr_forecast.data import first_diagnosis_labels, prepare_dataset, shift_years


def _csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture_data(root):
    _csv(root / "patient_splits.csv", ["Id", "split"], [
        {"Id": "a", "split": "train"}, {"Id": "b", "split": "val"}, {"Id": "c", "split": "test"}])
    _csv(root / "target_conditions.csv", ["CODE", "DESCRIPTION"], [
        {"CODE": "old", "DESCRIPTION": "old disease"}, {"CODE": "new", "DESCRIPTION": "new disease"}])
    _csv(root / "test_anchors.csv", ["Id", "anchor_date"], [{"Id": "c", "anchor_date": "2015-01-01"}])
    for folder, patients in (("train_val", ["a", "b"]), ("test", ["c"])):
        _csv(root / folder / "patients.csv", ["Id", "BIRTHDATE", "GENDER", "DEATHDATE"], [
            {"Id": p, "BIRTHDATE": "1980-01-01", "GENDER": "F", "DEATHDATE": "2021-01-01"} for p in patients])
        _csv(root / folder / "encounters.csv", ["PATIENT", "START", "STOP", "CODE"], [
            {"PATIENT": p, "START": "2010-01-01", "STOP": "2010-01-02", "CODE": "past"} for p in patients] +
            ([{"PATIENT": p, "START": "2020-01-01", "STOP": "", "CODE": "future"} for p in patients] if folder == "train_val" else []))
        _csv(root / folder / "conditions.csv", ["PATIENT", "START", "STOP", "CODE"], [
            {"PATIENT": p, "START": "2010-01-01", "CODE": "old", "STOP": "2030-01-01"} for p in patients] +
            ([{"PATIENT": p, "START": "2017-01-01", "CODE": "old"} for p in patients] +
             [{"PATIENT": p, "START": "2015-01-01", "CODE": "new"} for p in patients] if folder == "train_val" else []))
        _csv(root / folder / "observations.csv", ["PATIENT", "DATE", "CODE", "TYPE", "VALUE", "UNITS"], [
            {"PATIENT": p, "DATE": "2011-01-01", "CODE": "lab", "TYPE": "numeric", "VALUE": str(10 if p == "a" else 100000), "UNITS": "mg"}
            for p in patients] + ([
                {"PATIENT": "a", "DATE": "2019-01-01", "CODE": "lab", "TYPE": "numeric", "VALUE": "99999999", "UNITS": "mg"}
            ] if folder == "train_val" else []))
    return root


def test_first_ever_and_boundary():
    labels, prior = first_diagnosis_labels(
        {"old": date(2010, 1, 1), "new": date(2015, 1, 1), "late": date(2020, 1, 2)},
        date(2015, 1, 1), date(2020, 1, 1), ["old", "new", "late", "absent"])
    np.testing.assert_array_equal(labels, [0, 1, 0, 0])
    np.testing.assert_array_equal(prior, [1, 0, 0, 0])
    assert shift_years(date(2020, 2, 29), -5) == date(2015, 2, 28)


def test_prepare_prevents_future_and_validation_fitting(tmp_path):
    source = fixture_data(tmp_path / "source")
    output = tmp_path / "prepared"
    metadata = prepare_dataset(source, output, max_events=8, min_token_count=1, n_value_bins=4)
    train, val, test = [np.load(output / f"{name}.npz", allow_pickle=False) for name in ["train", "val", "test"]]
    vocab = json.loads((output / "vocabulary.json").read_text())["token_to_id"]
    preprocessing = json.loads((output / "preprocessing.json").read_text())
    assert metadata["split_counts"] == {"train": 1, "val": 1, "test": 1}
    np.testing.assert_array_equal(train["labels"], [[0, 1]])
    np.testing.assert_array_equal(train["prior_conditions"], [[1, 0]])
    assert "labels" not in test.files
    assert test["anchor_dates"].tolist() == ["2015-01-01"]
    assert not any("future" in token or "COND:new" in token for token in vocab)
    assert preprocessing["numeric_bin_edges"]["OBS:lab|UNIT=mg"] == [10.0]
    assert preprocessing["static"]["age_mean"] == pytest.approx(35.00068446)
    assert (train["time_days"][train["tokens"] > 3] > 0).all()
    # BOS must not disclose the next event's time during next-event training.
    for split_arrays in (train, val, test):
        np.testing.assert_array_equal(split_arrays["time_days"][:, 0], 0)
    assert preprocessing["bos_time_days"] == 0.0
    assert test["patient_ids"].tolist() == ["c"]
    assert "DEATHDATE" not in metadata["static_inputs"]


def test_validation_changes_do_not_change_fitted_artifacts(tmp_path):
    source = fixture_data(tmp_path / "source")
    prepare_dataset(source, tmp_path / "before", max_events=8, min_token_count=1)
    # Add a validation-only categorical code and extreme lab. Neither may fit
    # the vocabulary, quantiles, nor static scaler.
    observation_path = source / "train_val" / "observations.csv"
    with observation_path.open("a", encoding="utf-8") as handle:
        handle.write("b,2012-01-01,unique,numeric,999999999,mg\n")
    prepare_dataset(source, tmp_path / "after", max_events=8, min_token_count=1)
    for filename in ["vocabulary.json", "preprocessing.json"]:
        assert (tmp_path / "before" / filename).read_bytes() == (tmp_path / "after" / filename).read_bytes()


def test_duplicate_split_fails(tmp_path):
    source = fixture_data(tmp_path / "source")
    with (source / "patient_splits.csv").open("a", encoding="utf-8") as handle:
        handle.write("a,val\n")
    with pytest.raises(ValueError, match="Duplicate patient"):
        prepare_dataset(source, tmp_path / "out")


def test_missing_official_patient_fails(tmp_path):
    source = fixture_data(tmp_path / "source")
    _csv(source / "test" / "patients.csv", ["Id", "BIRTHDATE", "GENDER"], [])
    with pytest.raises(ValueError, match="missing 1 official IDs"):
        prepare_dataset(source, tmp_path / "out")


def test_empty_history_kept_and_bos_survives_truncation(tmp_path):
    source = fixture_data(tmp_path / "source")
    _csv(source / "test" / "conditions.csv", ["PATIENT", "START", "CODE"], [])
    _csv(source / "test" / "observations.csv", ["PATIENT", "DATE", "CODE", "TYPE", "VALUE", "UNITS"], [])
    _csv(source / "test" / "encounters.csv", ["PATIENT", "START", "STOP", "CODE"], [])
    prepare_dataset(source, tmp_path / "out", max_events=2, min_token_count=1)
    test = np.load(tmp_path / "out" / "test.npz", allow_pickle=False)
    train = np.load(tmp_path / "out" / "train.npz", allow_pickle=False)
    assert test["tokens"].tolist() == [[2, 0]]
    assert train["tokens"][0, 0] == 2
    assert train["tokens"][0, 1] != 0
    # The old diagnosis is truncated from the sequence but remains available to
    # the first-ever target rule and the prior-condition prediction mask.
    assert train["prior_conditions"][0, 0]
    assert test["patient_ids"].tolist() == ["c"]
