from logger_config import logger
import pandas as pd
from data_preprocessing.label_encoder import LabelEncoderWrapper
from data_preprocessing.one_hot_encoder import OneHotEncoderWrapper
from data_preprocessing.ordinal_encoder import OrdinalEncoderWrapper
from common.exceptions import DataEncodingError, ConfigValidationError


class DataEncoder:
    """Handles categorical feature encoding based on JSON configuration."""

    def __init__(self, config):
        """
        Initializes DataEncoder.

        Args:
            config (dict): Encoding configuration.
        """
        if not isinstance(config, dict):
            raise ConfigValidationError(
                "Invalid encoding configuration. Expected a dictionary."
            )

        self.df = None
        self.encoding_config = config.get("encoding", {})
        self.target_column = config.get("target_column", "churn")  # Default to "churn"

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
        if df is None or df.empty:
            raise DataEncodingError(
                "Input DataFrame is empty or None. Cannot perform encoding."
            )

        for column, encoding_info in self.encoding_config.items():
            if column == "_metadata":
                continue

            if column not in df.columns:
                logger.warning(
                    f"Column '{column}' not found in dataset. Ignoring column."
                )

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
                    raise DataEncodingError(
                        f"Unknown encoding method '{method}' for column '{column}'."
                    )

            except Exception as e:
                raise DataEncodingError(
                    f"Failed to encode column '{column}': {str(e)}"
                ) from e

        return df

    def _apply_mean_encoding(self, df: pd.DataFrame, column: str, target_column: str):
        """Applies mean (target) encoding to a categorical column."""
        if target_column not in df.columns:
            raise DataEncodingError(
                f"Target column '{target_column}' not found for mean encoding of '{column}'."
            )

        try:
            mean_encoding = df.groupby(column)[target_column].mean().to_dict()
            df[f"{column}_{target_column}"] = df[column].map(mean_encoding)
            logger.info(f"Applied mean encoding to column '{column}'")
        except Exception as e:
            raise DataEncodingError(
                f"Mean encoding failed for column '{column}': {str(e)}"
            ) from e

        return df

    def _apply_label_encoding(self, df: pd.DataFrame, column: str):
        """Applies label encoding to a categorical column."""
        try:
            encoder = LabelEncoderWrapper()
            df = encoder.encode(df, column)
            logger.info(f"Applied label encoding to column '{column}'")
        except Exception as e:
            raise DataEncodingError(
                f"Label encoding failed for column '{column}': {str(e)}"
            ) from e

        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame, column: str):
        """Applies one-hot encoding to a categorical column."""
        try:
            encoder = OneHotEncoderWrapper()
            df = encoder.encode(df, column)
            logger.info(f"Applied one-hot encoding to column '{column}'")
        except Exception as e:
            raise DataEncodingError(
                f"One-hot encoding failed for column '{column}': {str(e)}"
            ) from e

        return df

    def _apply_ordinal_encoding(self, df: pd.DataFrame, column: str, categories: list):
        """Applies ordinal encoding to a categorical column."""
        if not categories:
            raise DataEncodingError(
                f"No categories specified for ordinal encoding in column '{column}'."
            )

        try:
            encoder = OrdinalEncoderWrapper(categories)
            df = encoder.encode(df, column)
            logger.info(f"Applied ordinal encoding to column '{column}'")
        except Exception as e:
            raise DataEncodingError(
                f"Ordinal encoding failed for column '{column}': {str(e)}"
            ) from e

        return df
