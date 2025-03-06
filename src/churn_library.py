from data_preprocessing.data_cleaner import DataCleaner
from eda.eda_visualizer import EDAVisualizer
from data_preprocessing.data_encoder import DataEncoder
from logger_config import logger
import pandas as pd
from common.utils import check_args_paths


def import_data(pth):
    """Loads CSV into a DataFrame without modifications."""
    return pd.read_csv(pth)


def perform_eda(df):
    """Performs exploratory data analysis on cleaned data."""
    eda = EDAVisualizer(df)

    eda.plot_histogram("churn")
    eda.plot_histogram("age")

    eda.plot_bar_chart("marital_status")
    eda.plot_kde("total_transaction_count")


def encoder_helper(df, config_json_file):
    """Encodes categorical features based on target variable proportions."""
    # encode categorical features
    encoder = DataEncoder(config_json_file=config_json_file)
    df_encoded = encoder.encode(df)
    return df_encoded


def main():
    try:
        config_path, csv_path, result_path = check_args_paths(
            description="Clean a dataset.",
            config_help="Path to the JSON configuration file.",
            csv_help="Path to the input dataset CSV file.",
            result_help="Path to save the final processed dataset CSV file.",
        )
    except FileNotFoundError as e:
        logger.error(e)
        print(e)
        return

    # Load raw data
    df_raw = import_data(csv_path)

    # Clean data : rename columns, drop columns, fill missing values, remove empty rows, replace categorical values
    cleaner = DataCleaner(config_json_file=config_path)
    df_cleaned = cleaner.clean_data(
        df_raw, drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
    )

    perform_eda(df_cleaned)

    df_encoded = encoder_helper(df_cleaned, config_json_file=config_path)

    df_encoded.to_csv(result_path, index=False)


if __name__ == "__main__":
    main()
