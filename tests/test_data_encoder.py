"""
This module contains unit tests for the DataEncoder class, which handles encoding of categorical variables.
Author: Marc Ennaji
Date: 2025-03-01

Disabled the W0621 pylint warning, as it triggers a false positive when using fixtures (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint) (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint)

"""

# pylint: disable=W0621

import pytest
import pandas as pd
from data_preprocessing.data_encoder import DataEncoder
from common.exceptions import ConfigValidationError, DataEncodingError
from logger_config import get_logger


@pytest.fixture
def sample_dataframe_fixture():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "gender": ["Male", "Female", "Female"],
        "education": ["Graduate", "High School", "Graduate"],
        "marital_status": ["Single", "Married", "Single"],
        "income_bracket": ["Less than $40K", "$40K - $60K", "$60K - $80K"],
        "card_type": ["Blue", "Silver", "Gold"],
        "churn": [1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config_fixture():
    """Fixture to provide a sample configuration for testing."""
    return {
        "encoding": {
            "gender": {"method": "label"},
            "education": {"method": "one-hot"},
            "marital_status": {
                "method": "ordinal",
                "categories": ["Single", "Married"],
            },
            "income_bracket": {"method": "mean"},
            "card_type": {"method": "mean"},
        },
        "target_column": "churn",
    }


def test_init_with_invalid_config():
    """Test initialization with an invalid configuration."""
    test_name = "test_init_with_invalid_config"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        with pytest.raises(ConfigValidationError):
            DataEncoder(config="invalid_config")
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_init_with_none_config():
    """Test initialization with None configuration."""
    test_name = "test_init_with_none_config"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        with pytest.raises(ConfigValidationError):
            DataEncoder(config=None)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_encode_with_empty_dataframe():
    """Test encoding with an empty DataFrame."""
    test_name = "test_encode_with_empty_dataframe"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    encoder = DataEncoder(config={})
    empty_df = pd.DataFrame()
    try:
        with pytest.raises(DataEncodingError):
            encoder.encode(empty_df)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_encode_with_valid_dataframe(sample_dataframe_fixture, sample_config_fixture):
    """Test encoding with a valid DataFrame and configuration."""
    test_name = "test_encode_with_valid_dataframe"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    encoder = DataEncoder(config=sample_config_fixture)
    try:
        encoded_df = encoder.encode(sample_dataframe_fixture)
        assert "gender" in encoded_df.columns
        assert (
            "education_High School" in encoded_df.columns
        )  # we don't test education_Graduate because we use drop_first=True
        assert "marital_status" in encoded_df.columns
        assert "income_bracket_churn" in encoded_df.columns
        assert "card_type_churn" in encoded_df.columns
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_label_encoding(sample_dataframe_fixture):
    """Test label encoding for a specific column."""
    test_name = "test_label_encoding"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {"encoding": {"gender": {"method": "label"}}}
    encoder = DataEncoder(config=config)
    try:
        encoded_df = encoder.encode(sample_dataframe_fixture)
        assert "gender" in encoded_df.columns
        assert pd.api.types.is_integer_dtype(encoded_df["gender"])
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_one_hot_encoding(sample_dataframe_fixture):
    """Test one-hot encoding for a specific column."""
    test_name = "test_one_hot_encoding"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {"encoding": {"education": {"method": "one-hot"}}}
    encoder = DataEncoder(config=config)
    try:
        encoded_df = encoder.encode(sample_dataframe_fixture)
        assert (
            "education_High School" in encoded_df.columns
        )  # we don't test education_Graduate because we use drop_first=True
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_ordinal_encoding(sample_dataframe_fixture):
    """Test ordinal encoding for a specific column with categories."""
    test_name = "test_ordinal_encoding"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {
        "encoding": {
            "marital_status": {"method": "ordinal", "categories": ["Single", "Married"]}
        }
    }
    encoder = DataEncoder(config=config)
    try:
        encoded_df = encoder.encode(sample_dataframe_fixture)
        assert "marital_status" in encoded_df.columns
        assert pd.api.types.is_float_dtype(encoded_df["marital_status"])
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_ordinal_encoding_no_categories(sample_dataframe_fixture):
    """Test ordinal encoding for a specific column without categories."""
    test_name = "test_ordinal_encoding_no_categories"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {"encoding": {"marital_status": {"method": "ordinal"}}}
    encoder = DataEncoder(config=config)
    try:
        with pytest.raises(DataEncodingError):
            encoder.encode(sample_dataframe_fixture)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_mean_encoding(sample_dataframe_fixture):
    """Test mean encoding for specific columns."""
    test_name = "test_mean_encoding"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {
        "encoding": {
            "income_bracket": {"method": "mean"},
            "card_type": {"method": "mean"},
        },
        "target_column": "churn",
    }
    encoder = DataEncoder(config=config)
    try:
        encoded_df = encoder.encode(sample_dataframe_fixture)
        assert "income_bracket_churn" in encoded_df.columns
        assert "card_type_churn" in encoded_df.columns
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_apply_unknown_encoding_method(sample_dataframe_fixture):
    """Test applying an unknown encoding method."""
    test_name = "test_apply_unknown_encoding_method"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {"encoding": {"gender": {"method": "unknown"}}}
    encoder = DataEncoder(config=config)
    try:
        with pytest.raises(DataEncodingError):
            encoder.encode(sample_dataframe_fixture)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_encode_with_metadata_column(sample_dataframe_fixture):
    """Test encoding with metadata in the configuration."""
    test_name = "test_encode_with_metadata_column"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {
        "encoding": {
            "_metadata": {"description": "Metadata for encoding"},
            "gender": {"method": "label"},
        }
    }
    encoder = DataEncoder(config=config)
    try:
        encoded_df = encoder.encode(sample_dataframe_fixture)
        assert "gender" in encoded_df.columns
        assert pd.api.types.is_integer_dtype(encoded_df["gender"])
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_encode_with_no_encoding_config(sample_dataframe_fixture):
    """Test encoding with no encoding configuration."""
    test_name = "test_encode_with_no_encoding_config"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {"encoding": {}}
    encoder = DataEncoder(config=config)
    try:
        encoded_df = encoder.encode(sample_dataframe_fixture)
        assert encoded_df.equals(sample_dataframe_fixture)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_encode_with_invalid_target_column(sample_dataframe_fixture):
    """Test encoding with an invalid target column."""
    test_name = "test_encode_with_invalid_target_column"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    config = {
        "encoding": {"income_bracket": {"method": "mean"}},
        "target_column": "non_existent_target",
    }
    encoder = DataEncoder(config=config)
    try:
        with pytest.raises(DataEncodingError):
            encoder.encode(sample_dataframe_fixture)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise
