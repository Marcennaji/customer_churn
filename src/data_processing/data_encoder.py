from logger_config import logger
import pandas as pd
from config_loader import load_config
from data_processing.label_encoder import LabelEncoderWrapper
from data_processing.one_hot_encoder import OneHotEncoderWrapper
from data_processing.ordinal_encoder import OrdinalEncoderWrapper
from common.utils import check_args_paths


class DataEncoder:
    """Handles categorical feature encoding based on JSON configuration."""

    def __init__(self, config_json_file=None):
        self.config_json_file = config_json_file
        self.df = None
        self.column_mapping = {}
        self.value_replacements = {}

        if config_json_file:
            self._load_json_config()

    def _load_json_config(self):
        """Loads encoding config from JSON."""
        config = load_config(self.config_json_file)
        self.encoding_config = config.get("encoding", {})

    def encode(self, df: pd.DataFrame):
        """Encodes categorical columns based on JSON-specified encoding methods."""
        for column, encoding_info in self.encoding_config.items():
            method = encoding_info.get("method")
            categories = encoding_info.get("categories", [])

            if method == "label":
                encoder = LabelEncoderWrapper()
            elif method == "one-hot":
                encoder = OneHotEncoderWrapper()
            elif method == "ordinal":
                if not categories:
                    logger.warning(
                        f"No categories specified for ordinal encoding in column '{column}'. Skipping encoding."
                    )
                    continue
                encoder = OrdinalEncoderWrapper(categories)
            else:
                logger.warning(
                    f"Unknown encoding method '{method}' for column '{column}'. Skipping encoding."
                )
                continue

            df = encoder.encode(df, column)
        return df


def main():

    try:
        config_path, csv_path, result_path = check_args_paths(
            description="Encode a dataset using DataCleaner.",
            config_help="Path to the encoder JSON configuration file.",
            csv_help="Path to the input dataset CSV file.",
            result_help="Path to save the encoded dataset CSV file.",
        )
    except FileNotFoundError as e:
        logger.error(e)
        print(e)
        return

    df = pd.read_csv(csv_path)
    encoder = DataEncoder(config_json_file=config_path)
    cleaned_df = encoder.encode(df)
    cleaned_df.to_csv(result_path, index=False)
    logger.info(f"Encoded dataset saved to {result_path}")


if __name__ == "__main__":
    main()
