import logging
import pandas as pd
from encoding.label_encoder import LabelEncoderWrapper
from encoding.one_hot_encoder import OneHotEncoderWrapper
from encoding.ordinal_encoder import OrdinalEncoderWrapper


class EncoderManager:
    """Manages encoding methods dynamically based on JSON configuration."""

    def __init__(self, encoding_config=None):
        """
        Args:
            encoding_config (dict): Configuration specifying encoding methods per column.
        """
        self.encoding_config = encoding_config or {}

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
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
