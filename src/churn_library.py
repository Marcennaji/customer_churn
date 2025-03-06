from data_preprocessing.data_cleaner import DataCleaner
from data_preprocessing.data_explorer import DataExplorer
from data_preprocessing.data_encoder import DataEncoder
from config_loader import load_config
import pandas as pd
import os


def import_data(pth):
    """Loads CSV into a DataFrame without modifications."""
    return pd.read_csv(pth)


def perform_eda(df):
    """Performs exploratory data analysis on cleaned data."""
    explorer = DataExplorer()
    explorer.perform_eda(df)


def encoder_helper(df, category_lst, response):
    """Encodes categorical features based on target variable proportions."""
    # encoder = EncoderHelper()
    # return encoder.encode(df, category_lst, response)


def main():
    """Main pipeline to process data and apply transformations."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../config/config.json")
    config = load_config(config_path)

    csv_file_path = os.path.join(script_dir, config.get("data").get("raw_data_path"))
    processed_csv_file_path = os.path.join(
        script_dir, config.get("data").get("processed_data_path")
    )

    cleaner_json_file_path = os.path.join(script_dir, config.get("cleaner_config_path"))
    encoder_json_file_path = os.path.join(script_dir, config.get("encoder_config_path"))

    # Load raw data
    df_raw = import_data(csv_file_path)

    # Clean data : rename columns, drop columns, fill missing values, remove empty rows, replace categorical values
    cleaner = DataCleaner(config_json_file=cleaner_json_file_path)
    df_cleaned = cleaner.clean_data(
        df_raw, drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
    )

    # encode categorical features
    encoder = DataEncoder(config_json_file=encoder_json_file_path)
    df_encoded = encoder.encode(df_cleaned)

    # Save the processed dataset
    df_encoded.to_csv(processed_csv_file_path, index=False)


if __name__ == "__main__":
    main()
