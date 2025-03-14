"""
Test module for the DatasetSplitter class in the models module.
Author: Marc Ennaji
Date: 2025-03-01

Disabled the W0621 pylint warning, as it triggers a false positive when using fixtures (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint) (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint)

"""

# pylint: disable=W0621

import pytest
import pandas as pd
from models.data_splitter import DatasetSplitter
from common.exceptions import DataSplittingError, ConfigValidationError
from logger_config import get_logger

# =========================== FIXTURES =========================== #


@pytest.fixture
def sample_dataframe_fixture():
    """Returns a sample DataFrame for testing."""
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "churn": [0, 1, 0, 1, 0],  # Target column
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_config_fixture():
    """Returns a valid configuration dictionary for dataset splitting."""
    return {
        "default": {
            "feature_columns": ["feature1", "feature2"],
            "target_column": "churn",
            "test_size": 0.3,
            "random_state": 42,
            "target_type": "int",
        }
    }


@pytest.fixture
def dataset_splitter_fixture(sample_dataframe_fixture, valid_config_fixture):
    """Returns a DatasetSplitter instance with valid parameters."""
    return DatasetSplitter(
        sample_dataframe_fixture, valid_config_fixture, profile="default"
    )


# =========================== TESTS =========================== #


def test_split_valid_data(dataset_splitter_fixture):
    """Test splitting works correctly with valid data and config."""
    test_name = "test_split_valid_data"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        X_train, X_test, y_train, y_test = dataset_splitter_fixture.split()

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_missing_feature_columns(sample_dataframe_fixture, valid_config_fixture):
    """Test error handling when feature columns are missing."""
    test_name = "test_missing_feature_columns"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    invalid_config = valid_config_fixture.copy()
    invalid_config["default"]["feature_columns"] = ["non_existent_column"]

    try:
        with pytest.raises(DataSplittingError, match="Missing feature columns"):
            DatasetSplitter(
                sample_dataframe_fixture, invalid_config, profile="default"
            ).split()
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_missing_target_column(sample_dataframe_fixture, valid_config_fixture):
    """Test error handling when the target column is missing."""
    test_name = "test_missing_target_column"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    invalid_config = valid_config_fixture.copy()
    invalid_config["default"]["target_column"] = "non_existent_target"

    try:
        with pytest.raises(DataSplittingError, match="Target column .* not found"):
            DatasetSplitter(
                sample_dataframe_fixture, invalid_config, profile="default"
            ).split()
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_invalid_profile(sample_dataframe_fixture, valid_config_fixture):
    """Test error handling when an invalid profile is provided."""
    test_name = "test_invalid_profile"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        with pytest.raises(
            ConfigValidationError, match="Profile 'invalid_profile' not found"
        ):
            DatasetSplitter(
                sample_dataframe_fixture,
                valid_config_fixture,
                profile="invalid_profile",
            )
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_empty_dataframe(valid_config_fixture):
    """Test error handling when an empty DataFrame is provided."""
    test_name = "test_empty_dataframe"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    empty_df = pd.DataFrame()

    try:
        with pytest.raises(
            DataSplittingError, match="Input DataFrame is empty or None"
        ):
            DatasetSplitter(empty_df, valid_config_fixture, profile="default")
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_invalid_target_type_conversion(sample_dataframe_fixture, valid_config_fixture):
    """Test error handling when target type conversion fails."""
    test_name = "test_invalid_target_type_conversion"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    sample_dataframe_fixture["churn"] = [
        "yes",
        "no",
        "yes",
        "no",
        "yes",
    ]  # Non-integer values
    invalid_config = valid_config_fixture.copy()
    invalid_config["default"]["target_type"] = "int"  # Force integer conversion

    try:
        with pytest.raises(
            DataSplittingError, match="Could not convert target column .* to type int"
        ):
            DatasetSplitter(
                sample_dataframe_fixture, invalid_config, profile="default"
            ).split()
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_invalid_config_format(sample_dataframe_fixture):
    """Test error handling when configuration is not a dictionary."""
    test_name = "test_invalid_config_format"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    invalid_config = ["invalid_config_list"]

    try:
        with pytest.raises(
            ConfigValidationError, match="Invalid configuration. Expected a dictionary"
        ):
            DatasetSplitter(sample_dataframe_fixture, invalid_config, profile="default")
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_custom_test_size(sample_dataframe_fixture, valid_config_fixture):
    """Test if the dataset is split correctly with a custom test_size."""
    test_name = "test_custom_test_size"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    valid_config_fixture["default"]["test_size"] = 0.5
    dataset_splitter = DatasetSplitter(
        sample_dataframe_fixture, valid_config_fixture, profile="default"
    )
    try:
        X_train, X_test, _, _ = dataset_splitter.split()

        assert X_train.shape[0] == 2  # 50% of 5 rows = 2 train, 3 test
        assert X_test.shape[0] == 3
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_default_random_state(sample_dataframe_fixture, valid_config_fixture):
    """Test if the random_state is correctly applied."""
    test_name = "test_default_random_state"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    dataset_splitter1 = DatasetSplitter(
        sample_dataframe_fixture, valid_config_fixture, profile="default"
    )
    dataset_splitter2 = DatasetSplitter(
        sample_dataframe_fixture, valid_config_fixture, profile="default"
    )

    try:
        X_train1, _, y_train1, _ = dataset_splitter1.split()
        X_train2, _, y_train2, _ = dataset_splitter2.split()

        # Ensure the splits are identical due to same random_state
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_different_random_state(sample_dataframe_fixture, valid_config_fixture):
    """Test that changing random_state affects the split."""
    test_name = "test_different_random_state"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    valid_config_fixture["default"]["random_state"] = 100
    dataset_splitter1 = DatasetSplitter(
        sample_dataframe_fixture, valid_config_fixture, profile="default"
    )

    valid_config_fixture["default"]["random_state"] = 200
    dataset_splitter2 = DatasetSplitter(
        sample_dataframe_fixture, valid_config_fixture, profile="default"
    )

    try:
        X_train1, _, y_train1, _ = dataset_splitter1.split()
        X_train2, _, y_train2, _ = dataset_splitter2.split()

        # Ensure different random states produce different splits
        assert not X_train1.equals(X_train2)
        assert not y_train1.equals(y_train2)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise
