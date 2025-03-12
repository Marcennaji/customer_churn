"""
This module handles label encoding for the customer churn project.
Author: Marc Ennaji
Date: 2023-10-10
"""

import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .encoder_base import EncoderBase


class LabelEncoderWrapper(EncoderBase):
    """Encodes categorical variables using Label Encoding."""

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            df[column] = LabelEncoder().fit_transform(df[column])
            logging.info(f"Applied Label Encoding to '{column}'.")
        return df
