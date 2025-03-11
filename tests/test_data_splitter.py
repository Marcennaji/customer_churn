import pytest
import pandas as pd
from models.data_splitter import DatasetSplitter
from common.exceptions import DataSplittingError, ConfigValidationError


# =========================== FIXTURES =========================== #


@pytest.fixture
def sample_dataframe():
    """Returns a sample DataFrame for testing."""
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "churn": [0, 1, 0, 1, 0],  # Target column
    }
    return pd.DataFrame(data)


@pytest.fixture
def valid_config():
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
def dataset_splitter(sample_dataframe, valid_config):
    """Returns a DatasetSplitter instance with valid parameters."""
    return DatasetSplitter(sample_dataframe, valid_config, profile="default")


# =========================== TESTS =========================== #


def test_split_valid_data(dataset_splitter):
    """Test splitting works correctly with valid data and config."""
    X_train, X_test, y_train, y_test = dataset_splitter.split()

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0


def test_missing_feature_columns(sample_dataframe, valid_config):
    """Test error handling when feature columns are missing."""
    invalid_config = valid_config.copy()
    invalid_config["default"]["feature_columns"] = ["non_existent_column"]

    with pytest.raises(DataSplittingError, match="Missing feature columns"):
        DatasetSplitter(
            sample_dataframe,
            invalid_config,
            profile="default").split()


def test_missing_target_column(sample_dataframe, valid_config):
    """Test error handling when the target column is missing."""
    invalid_config = valid_config.copy()
    invalid_config["default"]["target_column"] = "non_existent_target"

    with pytest.raises(DataSplittingError, match="Target column .* not found"):
        DatasetSplitter(
            sample_dataframe,
            invalid_config,
            profile="default").split()


def test_invalid_profile(sample_dataframe, valid_config):
    """Test error handling when an invalid profile is provided."""
    with pytest.raises(
        ConfigValidationError, match="Profile 'invalid_profile' not found"
    ):
        DatasetSplitter(
            sample_dataframe,
            valid_config,
            profile="invalid_profile")


def test_empty_dataframe(valid_config):
    """Test error handling when an empty DataFrame is provided."""
    empty_df = pd.DataFrame()

    with pytest.raises(DataSplittingError, match="Input DataFrame is empty or None"):
        DatasetSplitter(empty_df, valid_config, profile="default")


def test_invalid_target_type_conversion(sample_dataframe, valid_config):
    """Test error handling when target type conversion fails."""
    sample_dataframe["churn"] = ["yes", "no",
                                 "yes", "no", "yes"]  # Non-integer values
    invalid_config = valid_config.copy()
    # Force integer conversion
    invalid_config["default"]["target_type"] = "int"

    with pytest.raises(
        DataSplittingError, match="Could not convert target column .* to type int"
    ):
        DatasetSplitter(
            sample_dataframe,
            invalid_config,
            profile="default").split()


def test_invalid_config_format(sample_dataframe):
    """Test error handling when configuration is not a dictionary."""
    invalid_config = ["invalid_config_list"]

    with pytest.raises(
        ConfigValidationError, match="Invalid configuration. Expected a dictionary"
    ):
        DatasetSplitter(sample_dataframe, invalid_config, profile="default")


def test_custom_test_size(sample_dataframe, valid_config):
    """Test if the dataset is split correctly with a custom test_size."""
    valid_config["default"]["test_size"] = 0.5
    dataset_splitter = DatasetSplitter(
        sample_dataframe, valid_config, profile="default"
    )
    X_train, X_test, y_train, y_test = dataset_splitter.split()

    assert X_train.shape[0] == 2  # 50% of 5 rows = 2 train, 3 test
    assert X_test.shape[0] == 3


def test_default_random_state(sample_dataframe, valid_config):
    """Test if the random_state is correctly applied."""
    dataset_splitter1 = DatasetSplitter(
        sample_dataframe, valid_config, profile="default"
    )
    dataset_splitter2 = DatasetSplitter(
        sample_dataframe, valid_config, profile="default"
    )

    X_train1, X_test1, y_train1, y_test1 = dataset_splitter1.split()
    X_train2, X_test2, y_train2, y_test2 = dataset_splitter2.split()

    # Ensure the splits are identical due to same random_state
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)


def test_different_random_state(sample_dataframe, valid_config):
    """Test that changing random_state affects the split."""
    valid_config["default"]["random_state"] = 100
    dataset_splitter1 = DatasetSplitter(
        sample_dataframe, valid_config, profile="default"
    )

    valid_config["default"]["random_state"] = 200
    dataset_splitter2 = DatasetSplitter(
        sample_dataframe, valid_config, profile="default"
    )

    X_train1, X_test1, y_train1, y_test1 = dataset_splitter1.split()
    X_train2, X_test2, y_train2, y_test2 = dataset_splitter2.split()

    # Ensure different random states produce different splits
    assert not X_train1.equals(X_train2)
    assert not y_train1.equals(y_train2)
