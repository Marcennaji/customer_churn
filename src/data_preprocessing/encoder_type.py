"""
This module provides an abstract base class for categorical encoders in the customer churn project.
Author: Marc Ennaji
Date: 2025-03-01
"""

import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder


class EncoderBase(ABC):
    """Abstract Base Class for categorical encoders."""

    @abstractmethod
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Encodes a specific column in the dataset."""

    def get_encoder_info(self):
        """Returns information about the encoder."""
        return f"Encoder: {self.__class__.__name__}"


class OneHotEncoderWrapper(EncoderBase):
    """Encodes categorical variables using One-Hot Encoding."""

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            encoded_data = self.encoder.fit_transform(df[[column]])
            encoded_df = pd.DataFrame(
                encoded_data, columns=self.encoder.get_feature_names_out([column])
            )

            df = df.drop(columns=[column]).reset_index(drop=True)
            df = pd.concat([df, encoded_df], axis=1)
            logging.info("Applied One-Hot Encoding to '%s'.", column)
        else:
            logging.warning("Column '%s' not found in DataFrame.", column)
        return df


class OrdinalEncoderWrapper(EncoderBase):
    """Encodes categorical variables using Ordinal Encoding with predefined category order."""

    def __init__(self, categories):
        self.categories = categories
        self.encoder = OrdinalEncoder(
            categories=[self.categories],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            df[column] = self.encoder.fit_transform(df[[column]])
            logging.info(
                "Applied Ordinal Encoding to '%s' with categories: %s.",
                column,
                self.categories,
            )
        else:
            logging.warning("Column '%s' not found in DataFrame.", column)
        return df


class LabelEncoderWrapper(EncoderBase):
    """Encodes categorical variables using Label Encoding."""

    def __init__(self):
        self.encoder = LabelEncoder()

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            df[column] = self.encoder.fit_transform(df[column])
            logging.info("Applied Label Encoding to '%s'.", column)
        else:
            logging.warning("Column '%s' not found in DataFrame.", column)
        return df
