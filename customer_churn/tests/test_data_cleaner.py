import pytest
import pandas as pd
from unittest.mock import Mock
from cleaning.data_cleaner import DataCleaner
from cleaning.categorical_processor import CategoricalProcessor


@pytest.fixture
def setup_data_cleaner():
    """Sets up test environment with a sample dataset."""
    sample_data = {
        "Unnamed: 0": [0, 1, 2],  # Unnamed column should be removed
        "CLIENTNUM": [123, 456, 789],  # Column to be dropped
        "attrition_status": [
            "Existing Customer",
            "Attrited Customer",
            "Existing Customer",
        ],
        "gender": ["M", "F", "M"],
        "income_bracket": ["$40K - $60K", "Less than $40K", "$60K - $80K"],
        "missing_values": [1, None, 3],
    }
    df = pd.DataFrame(sample_data)

    # Mocking the CategoricalProcessor to prevent actual encoding logic from interfering with tests
    mock_categorical_processor = Mock(spec=CategoricalProcessor)

    # Initializing DataCleaner without a real CSV file (we manually assign the DataFrame)
    cleaner = DataCleaner(
        csv_file=None, categorical_processor=mock_categorical_processor
    )
    cleaner.df = df.copy()  # Directly setting the DataFrame

    return cleaner


def test_drop_unnamed_first_column(setup_data_cleaner):
    """Test if unnamed first column is removed."""
    cleaner = setup_data_cleaner
    cleaner.drop_unnamed_first_column()
    assert "Unnamed: 0" not in cleaner.df.columns


def test_drop_columns(setup_data_cleaner):
    """Test if specified columns are correctly dropped."""
    cleaner = setup_data_cleaner
    cleaner.drop_columns(["CLIENTNUM"])
    assert "CLIENTNUM" not in cleaner.df.columns


def test_rename_columns(setup_data_cleaner):
    """Test if columns are correctly renamed based on mapping."""
    cleaner = setup_data_cleaner
    cleaner.column_mapping = {"attrition_status": "customer_status", "gender": "sex"}
    cleaner.rename_columns()
    assert "customer_status" in cleaner.df.columns
    assert "sex" in cleaner.df.columns
    assert "attrition_status" not in cleaner.df.columns
    assert "gender" not in cleaner.df.columns


def test_fill_missing_values_mean(setup_data_cleaner):
    """Test if missing values are filled using the mean strategy."""
    cleaner = setup_data_cleaner
    cleaner.fill_missing_values(strategy="mean")
    assert not cleaner.df["missing_values"].isna().any()


def test_fill_missing_values_median(setup_data_cleaner):
    """Test if missing values are filled using the median strategy."""
    cleaner = setup_data_cleaner
    cleaner.fill_missing_values(strategy="median")
    assert not cleaner.df["missing_values"].isna().any()


def test_fill_missing_values_constant(setup_data_cleaner):
    """Test if missing values are filled with a constant value."""
    cleaner = setup_data_cleaner
    cleaner.fill_missing_values(strategy="constant", fill_value=0)
    assert not cleaner.df["missing_values"].isna().any()
    assert cleaner.df["missing_values"].iloc[1] == 0


def test_remove_empty_rows(setup_data_cleaner):
    """Test if rows containing only NaN values are removed."""
    cleaner = setup_data_cleaner
    cleaner.df.loc[3] = [None] * len(cleaner.df.columns)  # Adding an empty row
    num_rows_before = len(cleaner.df)
    cleaner.remove_empty_rows()
    assert len(cleaner.df) == num_rows_before - 1


def test_log_dataset_info(setup_data_cleaner):
    """Test if dataset logging runs without errors."""
    cleaner = setup_data_cleaner
    try:
        cleaner.log_dataset_info()
    except Exception as e:
        pytest.fail(f"log_dataset_info() raised an unexpected exception: {e}")


def test_clean_data_calls_categorical_processor(setup_data_cleaner):
    """Test if `CategoricalProcessor` methods are called during `clean_data`."""
    cleaner = setup_data_cleaner
    cleaner.clean_data(
        drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
    )

    # Ensure categorical processor's methods were called
    cleaner.categorical_processor.replace_column_values.assert_called_once()
    cleaner.categorical_processor.encode_categorical.assert_called_once()


def test_clean_data_integrity(setup_data_cleaner):
    """Test if `clean_data` correctly processes the DataFrame."""
    cleaner = setup_data_cleaner
    cleaner.column_mapping = {"attrition_status": "customer_status", "gender": "sex"}
    df_cleaned = cleaner.clean_data(
        drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
    )

    assert "Unnamed: 0" not in df_cleaned.columns
    assert "CLIENTNUM" not in df_cleaned.columns
    assert "customer_status" in df_cleaned.columns
    assert "sex" in df_cleaned.columns
    assert (
        not df_cleaned["missing_values"].isna().any()
    )  # Ensure missing values are filled


def test_save_data(setup_data_cleaner):
    """Test if `save_data` runs without errors."""
    cleaner = setup_data_cleaner
    try:
        cleaner.save_data("test_output.csv")
    except Exception as e:
        pytest.fail(f"save_data() raised an unexpected exception: {e}")
