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
from logger_config import get_logger


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
    test_name = "test_init_with_invalid_config"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        with pytest.raises(DataValidationError):
            DataCleaner(config="invalid_config")
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_init_with_none_config():
    """Test initialization of DataCleaner with a None config."""
    test_name = "test_init_with_none_config"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        with pytest.raises(DataValidationError):
            DataCleaner(config=None)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_clean_data_with_empty_dataframe():
    """Test clean_data method with an empty DataFrame."""
    test_name = "test_clean_data_with_empty_dataframe"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    empty_df = pd.DataFrame()
    try:
        with pytest.raises(DataPreprocessingError):
            cleaner.clean_data(empty_df)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_clean_data_with_valid_dataframe(sample_dataframe, sample_config):
    """Test clean_data method with a valid DataFrame and config."""
    test_name = "test_clean_data_with_valid_dataframe"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config=sample_config)
    try:
        cleaned_df = cleaner.clean_data(
            sample_dataframe, drop_columns=["C"], fill_strategy="mean"
        )
        assert "Unnamed: 0" not in cleaned_df.columns
        assert "Alpha" in cleaned_df.columns
        assert "A" not in cleaned_df.columns
        assert cleaned_df["Alpha"].isna().sum() == 0
        assert len(cleaned_df) == 3
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_drop_unnamed_first_column(sample_dataframe):
    """Test the _drop_unnamed_first_column method."""
    test_name = "test_drop_unnamed_first_column"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        cleaner.drop_unnamed_first_column(sample_dataframe)
        assert "Unnamed: 0" not in sample_dataframe.columns
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_drop_columns(sample_dataframe):
    """Test the _drop_columns method."""
    test_name = "test_drop_columns"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        cleaner.drop_columns(sample_dataframe, drop_columns=["A"])
        assert "A" not in sample_dataframe.columns
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_rename_columns(sample_dataframe, sample_config):
    """Test the _rename_columns method."""
    test_name = "test_rename_columns"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config=sample_config)
    try:
        cleaner.rename_columns(sample_dataframe)
        assert "Alpha" in sample_dataframe.columns
        assert "A" not in sample_dataframe.columns
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_fill_missing_values_mean(sample_dataframe):
    """Test the _fill_missing_values method with the 'mean' strategy."""
    test_name = "test_fill_missing_values_mean"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        cleaner.fill_missing_values(sample_dataframe, strategy="mean")
        assert sample_dataframe["A"].isna().sum() == 0
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_fill_missing_values_median(sample_dataframe):
    """Test the _fill_missing_values method with the 'median' strategy."""
    test_name = "test_fill_missing_values_median"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        cleaner.fill_missing_values(sample_dataframe, strategy="median")
        assert sample_dataframe["A"].isna().sum() == 0
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_fill_missing_values_mode(sample_dataframe):
    """Test the _fill_missing_values method with the 'mode' strategy."""
    test_name = "test_fill_missing_values_mode"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        cleaner.fill_missing_values(sample_dataframe, strategy="mode")
        assert sample_dataframe["A"].isna().sum() == 0
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_fill_missing_values_constant(sample_dataframe):
    """Test the _fill_missing_values method with the 'constant' strategy."""
    test_name = "test_fill_missing_values_constant"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        cleaner.fill_missing_values(sample_dataframe, strategy="constant", fill_value=0)
        assert sample_dataframe["A"].isna().sum() == 0
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_fill_missing_values_invalid_strategy(sample_dataframe):
    """Test the _fill_missing_values method with an invalid strategy."""
    test_name = "test_fill_missing_values_invalid_strategy"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        with pytest.raises(DataPreprocessingError):
            cleaner.fill_missing_values(sample_dataframe, strategy="invalid")
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_replace_categorical_values(sample_dataframe, sample_config):
    """Test the _replace_categorical_values method."""
    test_name = "test_replace_categorical_values"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config=sample_config)
    try:
        cleaner.replace_categorical_values(sample_dataframe)
        assert sample_dataframe["B"].iloc[0] == "X"
        assert sample_dataframe["B"].iloc[1] == "Y"
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_remove_empty_rows(sample_dataframe):
    """Test the _remove_empty_rows method."""
    test_name = "test_remove_empty_rows"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        cleaner.remove_empty_rows(sample_dataframe)
        assert len(sample_dataframe) == 3
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_check_missing_columns(sample_dataframe):
    """Test the _check_missing_columns method."""
    test_name = "test_check_missing_columns"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    cleaner = DataCleaner(config={})
    try:
        missing_columns = cleaner.check_missing_columns(sample_dataframe, ["A", "D"])
        assert "A" in missing_columns
        assert "D" not in missing_columns
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise
