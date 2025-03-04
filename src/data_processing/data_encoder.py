import logging
import pandas as pd
from config_loader import load_config
from data_processing.label_encoder import LabelEncoderWrapper
from data_processing.one_hot_encoder import OneHotEncoderWrapper
from data_processing.ordinal_encoder import OrdinalEncoderWrapper


logger = logging.getLogger("DataEncoder")


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
