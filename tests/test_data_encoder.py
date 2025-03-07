import pytest
import pandas as pd
from src.data_preprocessing.data_encoder import DataEncoder
from common.exceptions import ConfigValidationError, DataEncodingError


@pytest.fixture
def sample_dataframe():
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
def sample_config():
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
    with pytest.raises(ConfigValidationError):
        DataEncoder(config="invalid_config")


def test_init_with_none_config():
    with pytest.raises(ConfigValidationError):
        DataEncoder(config=None)


def test_encode_with_empty_dataframe():
    encoder = DataEncoder(config={})
    empty_df = pd.DataFrame()
    with pytest.raises(DataEncodingError):
        encoder.encode(empty_df)


def test_encode_with_valid_dataframe(sample_dataframe, sample_config):
    encoder = DataEncoder(config=sample_config)
    encoded_df = encoder.encode(sample_dataframe)
    assert "gender" in encoded_df.columns
    assert (
        "education_High School" in encoded_df.columns
    )  # we don't test education_Graduate because we use drop_first=True
    assert "marital_status" in encoded_df.columns
    assert "income_bracket_churn" in encoded_df.columns
    assert "card_type_churn" in encoded_df.columns


def test_label_encoding(sample_dataframe):
    config = {"encoding": {"gender": {"method": "label"}}}
    encoder = DataEncoder(config=config)
    encoded_df = encoder.encode(sample_dataframe)
    assert "gender" in encoded_df.columns
    assert pd.api.types.is_integer_dtype(encoded_df["gender"])


def test_one_hot_encoding(sample_dataframe):
    config = {"encoding": {"education": {"method": "one-hot"}}}
    encoder = DataEncoder(config=config)
    encoded_df = encoder.encode(sample_dataframe)
    assert (
        "education_High School" in encoded_df.columns
    )  # we don't test education_Graduate because we use drop_first=True


def test_ordinal_encoding(sample_dataframe):
    config = {
        "encoding": {
            "marital_status": {"method": "ordinal", "categories": ["Single", "Married"]}
        }
    }
    encoder = DataEncoder(config=config)
    encoded_df = encoder.encode(sample_dataframe)
    assert "marital_status" in encoded_df.columns
    assert pd.api.types.is_float_dtype(encoded_df["marital_status"])


def test_ordinal_encoding_no_categories(sample_dataframe):
    config = {"encoding": {"marital_status": {"method": "ordinal"}}}
    encoder = DataEncoder(config=config)
    with pytest.raises(DataEncodingError):
        encoder.encode(sample_dataframe)


def test_mean_encoding(sample_dataframe):
    config = {
        "encoding": {
            "income_bracket": {"method": "mean"},
            "card_type": {"method": "mean"},
        },
        "target_column": "churn",
    }
    encoder = DataEncoder(config=config)
    encoded_df = encoder.encode(sample_dataframe)
    assert "income_bracket_churn" in encoded_df.columns
    assert "card_type_churn" in encoded_df.columns


def test_apply_unknown_encoding_method(sample_dataframe):
    config = {"encoding": {"gender": {"method": "unknown"}}}
    encoder = DataEncoder(config=config)
    with pytest.raises(DataEncodingError):
        encoder.encode(sample_dataframe)


def test_encode_with_metadata_column(sample_dataframe):
    config = {
        "encoding": {
            "_metadata": {"description": "Metadata for encoding"},
            "gender": {"method": "label"},
        }
    }
    encoder = DataEncoder(config=config)
    encoded_df = encoder.encode(sample_dataframe)
    assert "gender" in encoded_df.columns
    assert pd.api.types.is_integer_dtype(encoded_df["gender"])


def test_encode_with_no_encoding_config(sample_dataframe):
    config = {"encoding": {}}
    encoder = DataEncoder(config=config)
    encoded_df = encoder.encode(sample_dataframe)
    assert encoded_df.equals(sample_dataframe)


def test_encode_with_invalid_target_column(sample_dataframe):
    config = {
        "encoding": {"income_bracket": {"method": "mean"}},
        "target_column": "non_existent_target",
    }
    encoder = DataEncoder(config=config)
    with pytest.raises(DataEncodingError):
        encoder.encode(sample_dataframe)
