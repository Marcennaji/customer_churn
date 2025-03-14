"""
This module handles categorical feature encoding based on JSON configuration
for the customer churn project.
Author: Marc Ennaji
Date: 2025-03-01
"""

import pandas as pd

from data_preprocessing.encoder_type import (
    LabelEncoderWrapper,
    OneHotEncoderWrapper,
    OrdinalEncoderWrapper,
)
from common.exceptions import DataEncodingError, ConfigValidationError
from logger_config import get_logger


class DataEncoder:
    """Handles categorical feature encoding based on JSON configuration."""

    def __init__(self, config):
        """
        Initializes DataEncoder.

        Args:
            config (dict): Encoding configuration.
        """
        get_logger().info("Initializing DataEncoder")
        self.validate_config(config)
        self.df = None
        self.encoding_config = config.get("encoding", {})
        self.target_column = config.get("target_column", "churn")  # Default to "churn"

    def validate_config(self, config):
        """
        Validates the encoding configuration.

        Args:
            config (dict): Encoding configuration.

        Raises:
            ConfigValidationError: If the configuration is invalid.
        """
        get_logger().info("Validating encoding configuration")
        if not isinstance(config, dict):
            message = "Invalid encoding configuration. Expected a dictionary."
            get_logger().error(message)
            raise ConfigValidationError(message)

    def encode(self, df: pd.DataFrame):
        """
        Encodes categorical columns based on JSON configuration.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.

        Returns:
            pd.DataFrame: Encoded DataFrame.

        Raises:
            DataEncodingError: If encoding fails.
        """
        get_logger().info("Encoding dataset")
        if df is None or df.empty:
            message = "Input DataFrame is empty or None. Cannot perform encoding."
            get_logger().error(message)
            raise DataEncodingError(message)

        for column, encoding_info in self.encoding_config.items():
            if column == "_metadata":
                continue

            if column not in df.columns:
                get_logger().warning(
                    "Column '%s' not found in dataset. Ignoring column.", column
                )
                continue

            method = encoding_info.get("method")
            categories = encoding_info.get("categories", [])

            try:
                if method == "mean":
                    df = self._apply_mean_encoding(df, column, self.target_column)
                elif method == "label":
                    df = self._apply_label_encoding(df, column)
                elif method == "one-hot":
                    df = self._apply_one_hot_encoding(df, column)
                elif method == "ordinal":
                    df = self._apply_ordinal_encoding(df, column, categories)
                else:
                    message = (
                        f"Unknown encoding method '{method}' for column '{column}'."
                    )
                    get_logger().error(message)
                    raise DataEncodingError(message)

            except Exception as e:
                message = f"Failed to encode column '{column}': {str(e)}"
                get_logger().error(message)
                raise DataEncodingError(message) from e

        return df

    def _apply_mean_encoding(self, df: pd.DataFrame, column: str, target_column: str):
        """Applies mean (target) encoding to a categorical column."""
        get_logger().info("Applying mean encoding to column: %s", column)
        if target_column not in df.columns:
            message = f"Target column '{target_column}' not found for mean encoding of '{column}'."
            get_logger().error(message)
            raise DataEncodingError(message)

        try:
            mean_encoding = df.groupby(column)[target_column].mean().to_dict()
            df[f"{column}_{target_column}"] = df[column].map(mean_encoding)
            get_logger().info("Applied mean encoding to column '%s'", column)
        except Exception as e:
            message = f"Mean encoding failed for column '{column}': {str(e)}"
            get_logger().error(message)
            raise DataEncodingError(message) from e

        return df

    def _apply_label_encoding(self, df: pd.DataFrame, column: str):
        """Applies label encoding to a categorical column."""
        get_logger().info("Applying label encoding to column: %s", column)
        try:
            encoder = LabelEncoderWrapper()
            df = encoder.encode(df, column)
            get_logger().info("Applied label encoding to column '%s'", column)
        except Exception as e:
            message = f"Label encoding failed for column '{column}': {str(e)}"
            get_logger().error(message)
            raise DataEncodingError(message) from e

        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame, column: str):
        """Applies one-hot encoding to a categorical column."""
        get_logger().info("Applying one-hot encoding to column: %s", column)
        try:
            encoder = OneHotEncoderWrapper()
            df = encoder.encode(df, column)
            get_logger().info("Applied one-hot encoding to column '%s'", column)
        except Exception as e:
            message = f"One-hot encoding failed for column '{column}': {str(e)}"
            get_logger().error(message)
            raise DataEncodingError(message) from e

        return df

    def _apply_ordinal_encoding(self, df: pd.DataFrame, column: str, categories: list):
        """Applies ordinal encoding to a categorical column."""
        get_logger().info("Applying ordinal encoding to column: %s", column)
        if not categories:
            message = (
                f"No categories specified for ordinal encoding in column '{column}'."
            )
            get_logger().error(message)
            raise DataEncodingError(message)

        try:
            encoder = OrdinalEncoderWrapper(categories)
            df = encoder.encode(df, column)
            get_logger().info("Applied ordinal encoding to column '%s'", column)
        except Exception as e:
            message = f"Ordinal encoding failed for column '{column}': {str(e)}"
            get_logger().error(message)
            raise DataEncodingError(message) from e

        return df
