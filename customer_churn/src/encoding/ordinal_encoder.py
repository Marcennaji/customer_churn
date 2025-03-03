import pandas as pd
import logging
from sklearn.preprocessing import OrdinalEncoder
from .encoder_base import EncoderBase


class OrdinalEncoderWrapper(EncoderBase):
    """Encodes categorical variables using Ordinal Encoding with predefined category order."""

    def __init__(self, categories):
        self.categories = categories

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            encoder = OrdinalEncoder(
                categories=[self.categories],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            df[column] = encoder.fit_transform(df[[column]])
            logging.info(
                f"Applied Ordinal Encoding to '{column}' with categories: {self.categories}."
            )
        return df
