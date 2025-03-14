"""
This module handles general data cleaning operations for the customer churn project.
Author: Marc Ennaji
Date: 2025-03-01
"""

import pandas as pd
from logger_config import get_logger

from common.exceptions import (
    DataValidationError,
    DataPreprocessingError,
)

pd.set_option("future.no_silent_downcasting", True)


class DataCleaner:
    """Handles general data cleaning operations (decoupled from encoding)."""

    def __init__(self, config):
        """
        Initializes the DataCleaner.

        Args:
            config (dict): Configuration dictionary containing column mappings and value replacements.
        """
        get_logger().info("Initializing DataCleaner")
        if not isinstance(config, dict):
            message = "Invalid configuration provided. Expected a dictionary."
            get_logger().error(message)
            raise DataValidationError(message)

        self.df = None
        self.column_mapping = config.get("column_names", {}) or {}
        self.value_replacements = config.get("column_values", {}) or {}

    def clean_data(
        self,
        df,
        drop_columns=None,
        fill_strategy=None,
        fill_value=None,
        remove_empty=True,
    ):
        """Applies general cleaning operations to the dataset."""
        get_logger().info("Cleaning data")
        if df is None or df.empty:
            message = "Dataframe is empty or None. Cannot perform cleaning."
            get_logger().error(message)
            raise DataPreprocessingError(message)

        try:
            self.drop_unnamed_first_column(df)
            self.drop_columns(df, drop_columns)
            self.rename_columns(df)
            self.replace_categorical_values(df)
            self.log_dataset_info(df)

            if fill_strategy:
                self.fill_missing_values(
                    df, strategy=fill_strategy, fill_value=fill_value
                )

            if remove_empty:
                self.remove_empty_rows(df)

            get_logger().info("Data cleaning completed successfully.")
            return df
        except Exception as e:
            message = f"Error during data cleaning: {e}"
            get_logger().error(message)
            raise DataPreprocessingError(message) from e

    def drop_unnamed_first_column(self, df):
        """Drops the first column if it is unnamed."""
        get_logger().info("Dropping unnamed first column if present")
        if not df.empty and df.columns[0].startswith("Unnamed"):
            get_logger().info("Dropping unnamed first column: %s", df.columns[0])
            df.drop(df.columns[0], axis=1, inplace=True)

    def drop_columns(self, df, drop_columns=None):
        """Drops specified columns from the dataset."""
        get_logger().info("Dropping specified columns")
        if drop_columns:
            valid_columns = self.check_missing_columns(df, drop_columns)
            if valid_columns:
                df.drop(columns=valid_columns, inplace=True)
                get_logger().info("Dropped columns: %s", valid_columns)

    def rename_columns(self, df):
        """Renames dataset columns based on the JSON mapping."""
        get_logger().info("Renaming columns based on configuration")
        valid_columns = self.check_missing_columns(df, self.column_mapping.keys())
        columns_to_rename = {col: self.column_mapping[col] for col in valid_columns}
        if columns_to_rename:
            get_logger().info("Renaming columns: %s", columns_to_rename)
            df.rename(columns=columns_to_rename, inplace=True)

    def fill_missing_values(self, df, strategy="mean", fill_value=None):
        """
        Fills missing values based on a given strategy.

        Args:
            strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'constant').
            fill_value (any, optional): If strategy is 'constant', this value is required.
        """
        get_logger().info("Filling missing values using strategy: %s", strategy)
        if df.empty:
            message = "Cannot fill missing values on an empty dataset."
            get_logger().error(message)
            raise DataPreprocessingError(message)

        numerical_cols = df.select_dtypes(include=["number"]).columns

        if strategy == "mean":
            self.fill_mean(df, numerical_cols)
        elif strategy == "median":
            self.fill_median(df, numerical_cols)
        elif strategy == "mode":
            self.fill_mode(df, numerical_cols)
        elif strategy == "constant":
            if fill_value is None:
                message = "fill_value is required when strategy='constant'."
                get_logger().error(message)
                raise DataPreprocessingError(message)
            self.fill_constant(df, fill_value)
        else:
            message = f"Invalid fill strategy {strategy}. Choose from 'mean', 'median', 'mode', or 'constant'."
            get_logger().error(message)
            raise DataPreprocessingError(message)

    def fill_mean(self, df, numerical_cols):
        """Fills missing numerical values with the mean."""
        get_logger().info("Filling missing numerical values using mean")
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        get_logger().info("Filled missing numerical values using mean.")

    def fill_median(self, df, numerical_cols):
        """Fills missing numerical values with the median."""
        get_logger().info("Filling missing numerical values using median")
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        get_logger().info("Filled missing numerical values using median.")

    def fill_mode(self, df, numerical_cols):
        """Fills missing numerical values with the mode."""
        get_logger().info("Filling missing numerical values using mode")
        mode_values = (
            df[numerical_cols].dropna().mode().iloc[0]
            if not df[numerical_cols].mode().empty
            else None
        )
        df[numerical_cols] = df[numerical_cols].fillna(mode_values)
        get_logger().info("Filled missing numerical values using mode.")

    def fill_constant(self, df, fill_value):
        """Fills missing values with a constant value."""
        get_logger().info("Filling missing values with constant value: %s", fill_value)
        df.fillna(fill_value, inplace=True)
        get_logger().info("Filled missing values with constant value: %s", fill_value)

    def replace_categorical_values(self, df):
        """Replaces categorical values in the dataset based on predefined mappings."""
        get_logger().info("Replacing categorical values based on configuration")
        if df.empty:
            message = "Cannot replace categorical values on an empty dataset."
            get_logger().error(message)
            raise DataPreprocessingError(message)

        for column, replacements in self.value_replacements.items():
            if column in df.columns:
                df[column] = df[column].replace(replacements)
                get_logger().info(
                    "Replaced values in column '%s': %s", column, replacements
                )
            else:
                get_logger().warning(
                    "Column '%s' not found. Skipping replacements.", column
                )

    def remove_empty_rows(self, df):
        """Removes rows where all values are NaN."""
        get_logger().info("Removing empty rows")
        if df.empty:
            message = "Cannot remove empty rows from an empty dataset."
            get_logger().error(message)
            raise DataPreprocessingError(message)

        num_rows_before = len(df)
        df.dropna(how="all", inplace=True)
        num_rows_after = len(df)
        removed_count = num_rows_before - num_rows_after
        get_logger().info("Removed %d completely empty rows.", removed_count)

    def log_dataset_info(self, df):
        """Logs dataset shape, missing values per column, and missing rows."""
        get_logger().info("Logging dataset information")
        if df is not None:
            num_rows, num_cols = df.shape
            missing_values_per_column = df.isna().sum()
            columns_with_missing_values = missing_values_per_column[
                missing_values_per_column > 0
            ]

            get_logger().info("Dataset Shape: %d rows, %d columns", num_rows, num_cols)
            if not columns_with_missing_values.empty:
                get_logger().info(
                    "Columns with Missing Values:\n%s", columns_with_missing_values
                )
            else:
                get_logger().info("No columns with missing values.")

    def check_missing_columns(self, df, columns):
        """Checks if any columns are missing before performing operations."""
        get_logger().info("Checking for missing columns")
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            get_logger().warning(
                "Some specified columns were not found in the dataset: %s", missing_cols
            )
        return [col for col in columns if col in df.columns]
