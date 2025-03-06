from logger_config import logger
import pandas as pd
from config_loader import load_config
from data_preprocessing.data_cleaner import DataCleaner
from data_preprocessing.label_encoder import LabelEncoderWrapper
from data_preprocessing.one_hot_encoder import OneHotEncoderWrapper
from data_preprocessing.ordinal_encoder import OrdinalEncoderWrapper
from common.utils import check_args_paths


class DataEncoder:
    """Handles categorical feature encoding based on JSON configuration."""

    def __init__(self, config_json_file=None):
        self.config_json_file = config_json_file
        self.df = None
        self.encoding_config = {}

        if config_json_file:
            self._load_json_config()

    def encode(self, df: pd.DataFrame, target_column="churn"):
        """Encodes categorical columns based on JSON configuration."""
        for column, encoding_info in self.encoding_config.items():
            if column == "_metadata":
                continue
            method = encoding_info.get("method")
            categories = encoding_info.get("categories", [])

            if method == "mean":
                df = self._apply_mean_encoding(df, column, self.target_column)
            elif method == "label":
                df = self._apply_label_encoding(df, column)
            elif method == "one-hot":
                df = self._apply_one_hot_encoding(df, column)
            elif method == "ordinal":
                df = self._apply_ordinal_encoding(df, column, categories)
            else:
                logger.warning(
                    f"Unknown encoding method '{method}' for column '{column}'. Skipping encoding."
                )

        return df

    def _load_json_config(self):
        """Loads encoding configuration from a JSON file."""
        config = load_config(self.config_json_file)
        self.encoding_config = config.get("encoding", {})
        self.target_column = config.get("target_column", "churn")  # Default to "churn"

    def _apply_mean_encoding(self, df: pd.DataFrame, column: str, target_column: str):
        """Applies mean (target) encoding to a categorical column."""
        if target_column not in df.columns:
            logger.warning(
                f"Target column '{target_column}' not found. Skipping mean encoding for '{column}'."
            )
            return df

        mean_encoding = df.groupby(column)[target_column].mean().to_dict()
        df[f"{column}_{target_column}"] = df[column].map(mean_encoding)
        logger.info(f"Applied mean encoding to column '{column}'")
        return df

    def _apply_label_encoding(self, df: pd.DataFrame, column: str):
        """Applies label encoding to a categorical column."""
        encoder = LabelEncoderWrapper()
        df = encoder.encode(df, column)
        logger.info(f"Applied label encoding to column '{column}'")
        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame, column: str):
        """Applies one-hot encoding to a categorical column."""
        encoder = OneHotEncoderWrapper()
        df = encoder.encode(df, column)
        logger.info(f"Applied one-hot encoding to column '{column}'")
        return df

    def _apply_ordinal_encoding(self, df: pd.DataFrame, column: str, categories: list):
        """Applies ordinal encoding to a categorical column."""
        if not categories:
            logger.warning(
                f"No categories specified for ordinal encoding in column '{column}'. Skipping encoding."
            )
            return df

        encoder = OrdinalEncoderWrapper(categories)
        df = encoder.encode(df, column)
        logger.info(f"Applied ordinal encoding to column '{column}'")
        return df


def main():
    try:
        config_path, csv_path, result_path = check_args_paths(
            description="Clean and encode a dataset.",
            config_help="Path to the JSON configuration file.",
            csv_help="Path to the input dataset CSV file.",
            result_help="Path to save the final processed dataset CSV file.",
        )
    except FileNotFoundError as e:
        logger.error(e)
        print(e)
        return

    df = pd.read_csv(csv_path)

    cleaner = DataCleaner(config_json_file=config_path)
    cleaned_df = cleaner.clean_data(df)

    encoder = DataEncoder(config_json_file=config_path)
    encoded_df = encoder.encode(cleaned_df)

    encoded_df.to_csv(result_path, index=False)
    logger.info(f"Processed dataset saved to {result_path}")


if __name__ == "__main__":
    main()
