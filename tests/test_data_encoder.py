import pytest
import pandas as pd
import json
from data_processing.data_encoder import DataEncoder


@pytest.fixture
def sample_dataframe():
    data = {
        "A": ["cat", "dog", "fish"],
        "B": ["red", "blue", "green"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config(tmp_path):
    config_data = {
        "encoding": {
            "A": {"method": "label"},
            "B": {"method": "one-hot"},
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return str(config_file)


def test_load_json_config(sample_config):
    encoder = DataEncoder(config_json_file=sample_config)
    assert "A" in encoder.encoding_config
    assert encoder.encoding_config["A"]["method"] == "label"


def test_label_encoding(sample_dataframe, sample_config):
    encoder = DataEncoder(config_json_file=sample_config)
    encoded_df = encoder.encode(sample_dataframe)
    assert "A" in encoded_df.columns
    assert pd.api.types.is_integer_dtype(encoded_df["A"])


def test_one_hot_encoding(sample_dataframe, sample_config):
    print(sample_dataframe)
    encoder = DataEncoder(config_json_file=sample_config)
    encoded_df = encoder.encode(sample_dataframe)
    # we won't get B_blue column because of drop="first"
    assert "B_red" in encoded_df.columns
    assert "B_green" in encoded_df.columns


def test_unknown_encoding_method(sample_dataframe, tmp_path):
    config_data = {
        "encoding": {
            "A": {"method": "unknown"},
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    encoder = DataEncoder(config_json_file=str(config_file))
    encoded_df = encoder.encode(sample_dataframe)
    assert "A" in encoded_df.columns
    assert encoded_df["A"].dtype == "object"


def test_ordinal_encoding(sample_dataframe, tmp_path):
    config_data = {
        "encoding": {
            "A": {"method": "ordinal", "categories": ["cat", "dog", "fish"]},
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    encoder = DataEncoder(config_json_file=str(config_file))
    encoded_df = encoder.encode(sample_dataframe)
    assert "A" in encoded_df.columns
    assert pd.api.types.is_float_dtype(encoded_df["A"])
    assert encoded_df["A"].tolist() == [0, 1, 2]
