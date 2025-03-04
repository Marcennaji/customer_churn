from logger_config import logger
import pandas as pd
from config_loader import load_config
from data_processing.label_encoder import LabelEncoderWrapper
from data_processing.one_hot_encoder import OneHotEncoderWrapper
from data_processing.ordinal_encoder import OrdinalEncoderWrapper
import os
import argparse


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
                    logging.warning(
                        f"No categories specified for ordinal encoding in column '{column}'. Skipping encoding."
                    )
                    continue
                encoder = OrdinalEncoderWrapper(categories)
            else:
                logging.warning(
                    f"Unknown encoding method '{method}' for column '{column}'. Skipping encoding."
                )
                continue

            df = encoder.encode(df, column)
        return df


def check_paths():
    parser = argparse.ArgumentParser(description="Clean a dataset using DataEncoder.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the encoder JSON configuration file.",
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to the input dataset CSV file."
    )
    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="Path to save the encoded dataset CSV file.",
    )
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Check if CSV file exists
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    # Check if the directory of the result file exists
    result_dir = os.path.dirname(args.result)
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(
            f"Directory for result file does not exist: {result_dir}"
        )

    return args.config, args.csv, args.result


def main():

    try:
        config_path, csv_path, result_path = check_paths()
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
