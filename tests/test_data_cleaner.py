import pytest
import pandas as pd
from data_processing.data_cleaner import DataCleaner
import json


@pytest.fixture
def sample_dataframe():
    data = {
        "Unnamed: 0": [1, 2, 3],
        "A": [1, 2, None],
        "B": ["x", "y", "z"],
        "C": [None, None, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config(tmp_path):
    config_data = {
        "column_names": {"A": "Alpha", "B": "Beta"},
        "column_values": {"B": {"x": "X", "y": "Y"}},
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return str(config_file)


def test_drop_unnamed_first_column(sample_dataframe):
    cleaner = DataCleaner()
    cleaner._drop_unnamed_first_column(sample_dataframe)
    assert "Unnamed: 0" not in sample_dataframe.columns


def test_drop_columns(sample_dataframe):
    cleaner = DataCleaner()
    cleaner._drop_columns(sample_dataframe, drop_columns=["A"])
    assert "A" not in sample_dataframe.columns


def test_rename_columns(sample_dataframe, sample_config):
    cleaner = DataCleaner(config_json_file=sample_config)
    cleaner._rename_columns(sample_dataframe)
    assert "Alpha" in sample_dataframe.columns
    assert "A" not in sample_dataframe.columns


def test_fill_missing_values_mean(sample_dataframe):
    cleaner = DataCleaner()
    cleaner._fill_missing_values(sample_dataframe, strategy="mean")
    assert sample_dataframe["A"].isna().sum() == 0


def test_replace_categorical_values(sample_dataframe, sample_config):
    cleaner = DataCleaner(config_json_file=sample_config)
    cleaner._replace_categorical_values(sample_dataframe)
    assert sample_dataframe["B"].iloc[0] == "X"
    assert sample_dataframe["B"].iloc[1] == "Y"


def test_remove_empty_rows(sample_dataframe):
    cleaner = DataCleaner()
    cleaner._remove_empty_rows(sample_dataframe)
    assert len(sample_dataframe) == 3


def test_clean_data(sample_dataframe, sample_config):
    cleaner = DataCleaner(config_json_file=sample_config)
    cleaned_df = cleaner.clean_data(
        sample_dataframe, drop_columns=["C"], fill_strategy="mean"
    )
    assert "Unnamed: 0" not in cleaned_df.columns
    assert "Alpha" in cleaned_df.columns
    assert cleaned_df["Alpha"].isna().sum() == 0
    assert len(cleaned_df) == 3
