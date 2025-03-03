import pandas as pd
import logging
import os

from config_loader import load_config
from cleaning.categorical_processor import CategoricalProcessor

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


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
        self.categorical_processor = categorical_processor  # Injected dependency

        if cleaner_config_json_file:
            self.column_mapping, value_replacements, encoding_config = (
                self._load_json_config()
            )

            if self.categorical_processor:
                self.categorical_processor.value_replacements = value_replacements

    def _load_json_config(self):
        """Loads column mappings, categorical replacements, and encoding configuration from JSON."""
        config = load_config(self.cleaner_config_json_file)
        return (
            config.get("column_names", {}),
            config.get("column_values", {}),
            config.get("encoding", {}),
        )

    def load_data(self):
        """Loads the dataset from CSV."""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file '{self.csv_file}' not found.")

        self.df = pd.read_csv(self.csv_file)
        logging.info(f"Dataset '{self.csv_file}' loaded successfully.")

    def drop_unnamed_first_column(self):
        """Drops the first column if it is unnamed."""
        if not self.df.empty and self.df.columns[0].startswith("Unnamed"):
            logging.info(f"Dropping unnamed first column: {self.df.columns[0]}")
            self.df.drop(self.df.columns[0], axis=1, inplace=True)

    def check_missing_columns(self, columns):
        """Checks if any columns are missing before performing operations."""
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            logging.warning(
                f"Some specified columns were not found in the dataset: {missing_cols}"
            )
        return [
            col for col in columns if col in self.df.columns
        ]  # Returns only valid columns

    def drop_columns(self, drop_columns=None):
        """Drops specified columns from the dataset."""
        if drop_columns:
            valid_columns = self.check_missing_columns(drop_columns)
            if valid_columns:
                self.df.drop(columns=valid_columns, inplace=True)
                logging.info(f"Dropped columns: {valid_columns}")

    def rename_columns(self):
        """Renames dataset columns based on the JSON mapping."""
        if not self.df.empty and self.column_mapping:
            valid_columns = self.check_missing_columns(self.column_mapping.keys())
            columns_to_rename = {col: self.column_mapping[col] for col in valid_columns}
            if columns_to_rename:
                logging.info(f"Renaming columns: {columns_to_rename}")
                self.df.rename(columns=columns_to_rename, inplace=True)

    def fill_missing_values(self, strategy="mean", fill_value=None):
        """
        Fills missing values based on a given strategy.

        Args:
            strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'constant').
            fill_value (any, optional): If strategy is 'constant', this value is required.
        """
        if not self.df.empty:
            if strategy == "mean":
                self.df.fillna(self.df.mean(), inplace=True)
                logging.info("Filled missing values using mean.")
            elif strategy == "median":
                self.df.fillna(self.df.median(), inplace=True)
                logging.info("Filled missing values using median.")
            elif strategy == "mode":
                mode_values = (
                    self.df.dropna().mode().iloc[0]
                    if not self.df.mode().empty
                    else None
                )
                self.df.fillna(mode_values, inplace=True)
                logging.info("Filled missing values using mode.")
            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError("fill_value is required when strategy='constant'.")
                self.df.fillna(fill_value, inplace=True)
                logging.info(f"Filled missing values with constant value: {fill_value}")
            else:
                raise ValueError(
                    f"Invalid fill strategy '{strategy}'. Choose from 'mean', 'median', 'mode', or 'constant'."
                )

    def remove_empty_rows(self):
        """Removes rows where all values are NaN."""
        if not self.df.empty:
            num_rows_before = len(self.df)
            self.df.dropna(how="all", inplace=True)
            num_rows_after = len(self.df)
            removed_count = num_rows_before - num_rows_after
            logging.info(f"Removed {removed_count} completely empty rows.")

    def log_dataset_info(self):
        """Logs dataset shape, missing values per column, and missing rows."""
        if self.df is not None:
            num_rows, num_cols = self.df.shape
            missing_values_per_column = self.df.isna().sum()
            missing_values_per_row = self.df.isna().sum(axis=1)

            logging.info(f"Dataset Shape: {num_rows} rows, {num_cols} columns")
            logging.info(
                "Missing Values per Column:\n" + str(missing_values_per_column)
            )
            logging.info(
                "Missing Values per Row:\n"
                + str(missing_values_per_row.value_counts().sort_index())
            )

    def save_data(self, output_path):
        """Saves the cleaned dataset to a CSV file."""
        if self.df is not None and not self.df.empty:
            self.df.to_csv(output_path, index=False)
            logging.info(f"Cleaned dataset saved to {output_path}.")
        else:
            logging.warning("No data to save. Make sure to clean the data first.")

    def clean_data(
        self, drop_columns=None, fill_strategy=None, fill_value=None, remove_empty=True
    ):
        """Performs general cleaning and delegates categorical processing."""
        self.load_data()
        self.drop_unnamed_first_column()
        self.drop_columns(drop_columns)
        self.rename_columns()
        self.log_dataset_info()

        if fill_strategy:
            self.fill_missing_values(strategy=fill_strategy, fill_value=fill_value)

        if remove_empty:
            self.remove_empty_rows()

        if self.categorical_processor:
            self.categorical_processor.replace_column_values(self.df)
            self.df = self.categorical_processor.encode_categorical(self.df)

        logging.info("Data cleaning completed successfully.")
        return self.df


def main():
    # Example usage
    cleaner = DataCleaner(
        csv_file="data/bank_data.csv",
        cleaner_config_json_file="data/bank_data_cleaner_config.json",
        categorical_processor=CategoricalProcessor(),
    )
    cleaner.clean_data(
        drop_columns=["CLIENTNUM"], fill_strategy=None, remove_empty=True
    )
    cleaner.save_data("data/cleaned_dataset.csv")


if __name__ == "__main__":
    main()
