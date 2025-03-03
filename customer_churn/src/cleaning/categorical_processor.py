import logging
import pandas as pd
from encoding.encoder_manager import EncoderManager


class CategoricalProcessor:
    """Handles categorical value replacements and encoding based on JSON configuration."""

    def __init__(self, value_replacements=None, encoder_manager=None):
        """
        Initializes the CategoricalProcessor.

        Args:
            value_replacements (dict): Mapping for categorical replacements.
            encoder_manager (EncoderManager): Manages encoding dynamically per column.
        """
        self.value_replacements = value_replacements or {}
        self.encoder_manager = encoder_manager or EncoderManager()

    def replace_column_values(self, df: pd.DataFrame):
        """Replaces categorical values in the dataset based on predefined mappings."""
        if not df.empty and self.value_replacements:
            for column, replacements in self.value_replacements.items():
                if column in df.columns:
                    df[column] = df[column].replace(replacements)
                    logging.info(
                        f"Replaced values in column '{column}': {replacements}"
                    )
                else:
                    logging.warning(
                        f"Column '{column}' not found in dataset. Skipping replacements."
                    )
        return df

    def encode_categorical(self, df: pd.DataFrame):
        """Encodes categorical variables using EncoderManager."""
        if self.encoder_manager:
            df = self.encoder_manager.encode(df)
        return df
