import pandas as pd
from logger_config import logger
import os

from config_loader import load_config
from cleaning.categorical_processor import CategoricalProcessor
from encoding.encoder_manager import EncoderManager


class DataCleaner:
    """Handles general data cleaning while delegating categorical processing to CategoricalProcessor."""

    def __init__(
        self, csv_file, cleaner_config_json_file=None, categorical_processor=None
    ):
        """
        Initializes the DataCleaner.

        Args:
            csv_file (str): Path to the dataset CSV file.
            cleaner_config_json_file (str, optional): Path to the JSON configuration.
            categorical_processor (CategoricalProcessor, optional): Handles categorical value processing.
        """
        self.csv_file = csv_file
        self.cleaner_config_json_file = cleaner_config_json_file
        self.df = None
        self.column_mapping = {}
        self.categorical_processor = categorical_processor

        if cleaner_config_json_file:
            self.column_mapping, value_replacements, encoding_config = (
                self._load_json_config()
            )

            if self.categorical_processor:
                self.categorical_processor.value_replacements = value_replacements
                self.categorical_processor.encoder_manager = EncoderManager(
                    encoding_config
                )

    def clean_data(
        self, drop_columns=None, fill_strategy=None, fill_value=None, remove_empty=True
    ):
        """Performs general cleaning and delegates categorical processing."""
        self._load_data()
        self._drop_unnamed_first_column()
        self._drop_columns(drop_columns)
        self._rename_columns()
        self._log_dataset_info()

        if fill_strategy:
            self._fill_missing_values(strategy=fill_strategy, fill_value=fill_value)

        if remove_empty:
            self._remove_empty_rows()

        if self.categorical_processor:
            self.categorical_processor.replace_column_values(self.df)
            self.df = self.categorical_processor.encode_categorical(self.df)

        logger.info("Data cleaning completed successfully.")
        return self.df

    def save_data(self, output_path):
        """Saves the cleaned dataset to a CSV file."""
        if self.df is not None and not self.df.empty:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Cleaned dataset saved to {output_path}.")
        else:
            logger.warning("No data to save. Make sure to clean the data first.")

    def _load_data(self):
        """Loads the dataset from CSV."""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file '{self.csv_file}' not found.")

        self.df = pd.read_csv(self.csv_file)
        logger.info(f"Dataset '{self.csv_file}' loaded successfully.")

    def _load_json_config(self):
        """Loads column mappings, categorical replacements, and encoding configuration from JSON."""
        config = load_config(self.cleaner_config_json_file)
        return (
            config.get("column_names", {}),
            config.get("column_values", {}),
            config.get("encoding", {}),
        )

    def _drop_unnamed_first_column(self):
        """Drops the first column if it is unnamed."""
        if not self.df.empty and self.df.columns[0].startswith("Unnamed"):
            logger.info(f"Dropping unnamed first column: {self.df.columns[0]}")
            self.df.drop(self.df.columns[0], axis=1, inplace=True)

    def _drop_columns(self, drop_columns=None):
        """Drops specified columns from the dataset."""
        if drop_columns:
            valid_columns = self._check_missing_columns(drop_columns)
            if valid_columns:
                self.df.drop(columns=valid_columns, inplace=True)
                logger.info(f"Dropped columns: {valid_columns}")

    def _rename_columns(self):
        """Renames dataset columns based on the JSON mapping."""
        if not self.df.empty and self.column_mapping:
            valid_columns = self._check_missing_columns(self.column_mapping.keys())
            columns_to_rename = {col: self.column_mapping[col] for col in valid_columns}
            if columns_to_rename:
                logger.info(f"Renaming columns: {columns_to_rename}")
                self.df.rename(columns=columns_to_rename, inplace=True)

    def _fill_missing_values(self, strategy="mean", fill_value=None):
        """
        Fills missing values based on a given strategy.

        Args:
            strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'constant').
            fill_value (any, optional): If strategy is 'constant', this value is required.
        """
        if not self.df.empty:
            numerical_cols = self.df.select_dtypes(include=["number"]).columns

            if strategy == "mean":
                self.df[numerical_cols] = self.df[numerical_cols].fillna(
                    self.df[numerical_cols].mean()
                )
                logger.info("Filled missing numerical values using mean.")
            elif strategy == "median":
                self.df[numerical_cols] = self.df[numerical_cols].fillna(
                    self.df[numerical_cols].median()
                )
                logger.info("Filled missing numerical values using median.")
            elif strategy == "mode":
                mode_values = (
                    self.df[numerical_cols].dropna().mode().iloc[0]
                    if not self.df[numerical_cols].mode().empty
                    else None
                )
                self.df[numerical_cols] = self.df[numerical_cols].fillna(mode_values)
                logger.info("Filled missing numerical values using mode.")
            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError("fill_value is required when strategy='constant'.")
                self.df.fillna(fill_value, inplace=True)
                logger.info(f"Filled missing values with constant value: {fill_value}")
            else:
                raise ValueError(
                    f"Invalid fill strategy '{strategy}'. Choose from 'mean', 'median', 'mode', or 'constant'."
                )

    def _remove_empty_rows(self):
        """Removes rows where all values are NaN."""
        if not self.df.empty:
            num_rows_before = len(self.df)
            self.df.dropna(how="all", inplace=True)
            num_rows_after = len(self.df)
            removed_count = num_rows_before - num_rows_after
            logger.info(f"Removed {removed_count} completely empty rows.")

    def _log_dataset_info(self):
        """Logs dataset shape, missing values per column, and missing rows."""
        if self.df is not None:
            num_rows, num_cols = self.df.shape
            missing_values_per_column = self.df.isna().sum()
            columns_with_missing_values = missing_values_per_column[
                missing_values_per_column > 0
            ]

            missing_values_per_row = self.df.isna().sum(axis=1)
            rows_with_missing_values = self.df[missing_values_per_row > 0]

            logger.info(f"Dataset Shape: {num_rows} rows, {num_cols} columns")
            if not columns_with_missing_values.empty:
                logger.info(
                    "Columns with Missing Values:\n" + str(columns_with_missing_values)
                )
            else:
                logger.info("No columns with missing values.")

            if not rows_with_missing_values.empty:
                logger.info(
                    "Rows with Missing Values:\n" + str(rows_with_missing_values)
                )
            else:
                logger.info("No rows with missing values.")

    def _check_missing_columns(self, columns):
        """Checks if any columns are missing before performing operations."""
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            logger.warning(
                f"Some specified columns were not found in the dataset: {missing_cols}"
            )
        return [col for col in columns if col in self.df.columns]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "../../data/bank_data.csv")
    cleaned_csv_file_path = os.path.join(script_dir, "../../data/bank_data_cleaned.csv")
    json_file_path = os.path.join(
        script_dir, "../../config/bank_data_cleaner_config.json"
    )

    cleaner = DataCleaner(
        csv_file=csv_file_path,
        cleaner_config_json_file=json_file_path,
        categorical_processor=CategoricalProcessor(),
    )
    cleaner.clean_data(
        drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
    )
    cleaner.save_data(cleaned_csv_file_path)


if __name__ == "__main__":
    main()
