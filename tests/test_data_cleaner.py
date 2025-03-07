import pytest
import pandas as pd
from data_preprocessing.data_cleaner import DataCleaner
from common.exceptions import DataValidationError, DataPreprocessingError


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
def sample_config():
    return {
        "column_names": {"A": "Alpha", "B": "Beta"},
        "column_values": {"Beta": {"x": "X", "y": "Y"}},
    }


def test_init_with_invalid_config():
    with pytest.raises(DataValidationError):
        DataCleaner(config="invalid_config")


def test_init_with_none_config():
    cleaner = DataCleaner(config=None)
    assert cleaner.column_mapping == {}
    assert cleaner.value_replacements == {}


def test_clean_data_with_empty_dataframe():
    cleaner = DataCleaner(config={})
    empty_df = pd.DataFrame()
    with pytest.raises(DataPreprocessingError):
        cleaner.clean_data(empty_df)


def test_clean_data_with_valid_dataframe(sample_dataframe, sample_config):
    cleaner = DataCleaner(config=sample_config)
    cleaned_df = cleaner.clean_data(
        sample_dataframe, drop_columns=["C"], fill_strategy="mean"
    )
    assert "Unnamed: 0" not in cleaned_df.columns
    assert "Alpha" in cleaned_df.columns
    assert "A" not in cleaned_df.columns
    assert cleaned_df["Alpha"].isna().sum() == 0
    assert len(cleaned_df) == 3


def test_drop_unnamed_first_column(sample_dataframe):
    cleaner = DataCleaner(config={})
    cleaner._drop_unnamed_first_column(sample_dataframe)
    assert "Unnamed: 0" not in sample_dataframe.columns


def test_drop_columns(sample_dataframe):
    cleaner = DataCleaner(config={})
    cleaner._drop_columns(sample_dataframe, drop_columns=["A"])
    assert "A" not in sample_dataframe.columns


def test_rename_columns(sample_dataframe, sample_config):
    cleaner = DataCleaner(config=sample_config)
    cleaner._rename_columns(sample_dataframe)
    assert "Alpha" in sample_dataframe.columns
    assert "A" not in sample_dataframe.columns


def test_fill_missing_values_mean(sample_dataframe):
    cleaner = DataCleaner(config={})
    cleaner._fill_missing_values(sample_dataframe, strategy="mean")
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_median(sample_dataframe):
    cleaner = DataCleaner(config={})
    cleaner._fill_missing_values(sample_dataframe, strategy="median")
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_mode(sample_dataframe):
    cleaner = DataCleaner(config={})
    cleaner._fill_missing_values(sample_dataframe, strategy="mode")
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_constant(sample_dataframe):
    cleaner = DataCleaner(config={})
    cleaner._fill_missing_values(sample_dataframe, strategy="constant", fill_value=0)
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_invalid_strategy(sample_dataframe):
    cleaner = DataCleaner(config={})
    with pytest.raises(DataPreprocessingError):
        cleaner._fill_missing_values(sample_dataframe, strategy="invalid")


def test_replace_categorical_values(sample_dataframe, sample_config):
    cleaner = DataCleaner(config=sample_config)
    cleaner._replace_categorical_values(sample_dataframe)
    assert sample_dataframe["B"].iloc[0] == "X"
    assert sample_dataframe["B"].iloc[1] == "Y"


def test_remove_empty_rows(sample_dataframe):
    cleaner = DataCleaner(config={})
    cleaner._remove_empty_rows(sample_dataframe)
    assert len(sample_dataframe) == 2


def test_log_dataset_info(sample_dataframe, caplog):
    cleaner = DataCleaner(config={})
    cleaner._log_dataset_info(sample_dataframe)
    assert "Dataset Shape" in caplog.text
    assert "Columns with Missing Values" in caplog.text


def test_check_missing_columns(sample_dataframe):
    cleaner = DataCleaner(config={})
    missing_columns = cleaner._check_missing_columns(sample_dataframe, ["A", "D"])
    assert "A" in missing_columns
    assert "D" not in missing_columns
