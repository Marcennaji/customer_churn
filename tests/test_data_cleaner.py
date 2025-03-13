"""
Test module for the DataCleaner class in the data_preprocessing module.
Author: Marc Ennaji
Date: 2025-03-01

Disabled the W0621 pylint warning, as it triggers a false positive when using fixtures (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint) (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint)

"""

# pylint: disable=W0621

import pytest
import pandas as pd
from data_preprocessing.data_cleaner import DataCleaner
from common.exceptions import DataValidationError, DataPreprocessingError


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing."""
    data = {
        "Unnamed: 0": [1, 2, 3],
        "A": [1, 2, None],
        "B": ["x", "y", "z"],
        "C": [None, None, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration dictionary for testing."""
    return {
        "column_names": {"A": "Alpha", "B": "Beta"},
        "column_values": {"B": {"x": "X", "y": "Y"}},
    }


def test_init_with_invalid_config():
    """Test initialization of DataCleaner with an invalid config."""
    with pytest.raises(DataValidationError):
        DataCleaner(config="invalid_config")


def test_init_with_none_config():
    """Test initialization of DataCleaner with a None config."""
    with pytest.raises(DataValidationError):
        DataCleaner(config=None)


def test_clean_data_with_empty_dataframe():
    """Test clean_data method with an empty DataFrame."""
    cleaner = DataCleaner(config={})
    empty_df = pd.DataFrame()
    with pytest.raises(DataPreprocessingError):
        cleaner.clean_data(empty_df)


def test_clean_data_with_valid_dataframe(sample_dataframe, sample_config):
    """Test clean_data method with a valid DataFrame and config."""
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
    """Test the _drop_unnamed_first_column method."""
    cleaner = DataCleaner(config={})
    cleaner.drop_unnamed_first_column(sample_dataframe)
    assert "Unnamed: 0" not in sample_dataframe.columns


def test_drop_columns(sample_dataframe):
    """Test the _drop_columns method."""
    cleaner = DataCleaner(config={})
    cleaner.drop_columns(sample_dataframe, drop_columns=["A"])
    assert "A" not in sample_dataframe.columns


def test_rename_columns(sample_dataframe, sample_config):
    """Test the _rename_columns method."""
    cleaner = DataCleaner(config=sample_config)
    cleaner.rename_columns(sample_dataframe)
    assert "Alpha" in sample_dataframe.columns
    assert "A" not in sample_dataframe.columns


def test_fill_missing_values_mean(sample_dataframe):
    """Test the _fill_missing_values method with the 'mean' strategy."""
    cleaner = DataCleaner(config={})
    cleaner.fill_missing_values(sample_dataframe, strategy="mean")
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_median(sample_dataframe):
    """Test the _fill_missing_values method with the 'median' strategy."""
    cleaner = DataCleaner(config={})
    cleaner.fill_missing_values(sample_dataframe, strategy="median")
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_mode(sample_dataframe):
    """Test the _fill_missing_values method with the 'mode' strategy."""
    cleaner = DataCleaner(config={})
    cleaner.fill_missing_values(sample_dataframe, strategy="mode")
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_constant(sample_dataframe):
    """Test the _fill_missing_values method with the 'constant' strategy."""
    cleaner = DataCleaner(config={})
    cleaner.fill_missing_values(sample_dataframe, strategy="constant", fill_value=0)
    assert sample_dataframe["A"].isna().sum() == 0


def test_fill_missing_values_invalid_strategy(sample_dataframe):
    """Test the _fill_missing_values method with an invalid strategy."""
    cleaner = DataCleaner(config={})
    with pytest.raises(DataPreprocessingError):
        cleaner.fill_missing_values(sample_dataframe, strategy="invalid")


def test_replace_categorical_values(sample_dataframe, sample_config):
    """Test the _replace_categorical_values method."""
    cleaner = DataCleaner(config=sample_config)
    cleaner.replace_categorical_values(sample_dataframe)
    assert sample_dataframe["B"].iloc[0] == "X"
    assert sample_dataframe["B"].iloc[1] == "Y"


def test_remove_empty_rows(sample_dataframe):
    """Test the _remove_empty_rows method."""
    cleaner = DataCleaner(config={})
    cleaner.remove_empty_rows(sample_dataframe)
    assert len(sample_dataframe) == 3


def test_check_missing_columns(sample_dataframe):
    """Test the _check_missing_columns method."""
    cleaner = DataCleaner(config={})
    missing_columns = cleaner.check_missing_columns(sample_dataframe, ["A", "D"])
    assert "A" in missing_columns
    assert "D" not in missing_columns
