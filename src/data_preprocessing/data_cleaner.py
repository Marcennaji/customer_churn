from logger_config import logger
import pandas as pd
from common.utils import check_args_paths

pd.set_option("future.no_silent_downcasting", True)


class DataCleaner:
    """Handles general data cleaning operations (decoupled from encoding)."""

    def __init__(self, config):
        """
        Initializes the DataCleaner.

        Args:
            csv_file (str, optional): Path to the dataset CSV file.
            config_json_file (str, optional): Path to the JSON configuration.
        """

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
        self._drop_unnamed_first_column(df)
        self._drop_columns(df, drop_columns)
        self._rename_columns(df)
        self._replace_categorical_values(df)
        self._log_dataset_info(df)

        if fill_strategy:
            self._fill_missing_values(df, strategy=fill_strategy, fill_value=fill_value)

        if remove_empty:
            self._remove_empty_rows(df)

        logger.info("Data cleaning completed successfully.")
        return df

    def _drop_unnamed_first_column(self, df):
        """Drops the first column if it is unnamed."""
        if not df.empty and df.columns[0].startswith("Unnamed"):
            logger.info(f"Dropping unnamed first column: {df.columns[0]}")
            df.drop(df.columns[0], axis=1, inplace=True)

    def _drop_columns(self, df, drop_columns=None):
        """Drops specified columns from the dataset."""
        if drop_columns:
            valid_columns = self._check_missing_columns(df, drop_columns)
            if valid_columns:
                df.drop(columns=valid_columns, inplace=True)
                logger.info(f"Dropped columns: {valid_columns}")

    def _rename_columns(self, df):
        """Renames dataset columns based on the JSON mapping."""
        valid_columns = self._check_missing_columns(df, self.column_mapping.keys())
        columns_to_rename = {col: self.column_mapping[col] for col in valid_columns}
        if columns_to_rename:
            logger.info(f"Renaming columns: {columns_to_rename}")
            df.rename(columns=columns_to_rename, inplace=True)

    def _fill_missing_values(self, df, strategy="mean", fill_value=None):
        """
        Fills missing values based on a given strategy.

        Args:
            strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'constant').
            fill_value (any, optional): If strategy is 'constant', this value is required.
        """
        if not df.empty:
            numerical_cols = df.select_dtypes(include=["number"]).columns

            if strategy == "mean":
                self._fill_mean(df, numerical_cols)
            elif strategy == "median":
                self._fill_median(df, numerical_cols)
            elif strategy == "mode":
                self._fill_mode(df, numerical_cols)
            elif strategy == "constant":
                self._fill_constant(df, fill_value)
            else:
                raise ValueError(
                    f"Invalid fill strategy '{strategy}'. Choose from 'mean', 'median', 'mode', or 'constant'."
                )

    def _fill_mean(self, df, numerical_cols):
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        logger.info("Filled missing numerical values using mean.")

    def _fill_median(self, df, numerical_cols):
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        logger.info("Filled missing numerical values using median.")

    def _fill_mode(self, df, numerical_cols):
        mode_values = (
            df[numerical_cols].dropna().mode().iloc[0]
            if not df[numerical_cols].mode().empty
            else None
        )
        df[numerical_cols] = df[numerical_cols].fillna(mode_values)
        logger.info("Filled missing numerical values using mode.")

    def _fill_constant(self, df, fill_value):
        if fill_value is None:
            raise ValueError("fill_value is required when strategy='constant'.")
        df.fillna(fill_value, inplace=True)
        logger.info(f"Filled missing values with constant value: {fill_value}")

    def _replace_categorical_values(self, df):
        """Replaces categorical values in the dataset based on predefined mappings."""
        if not df.empty and self.value_replacements:
            for column, replacements in self.value_replacements.items():
                if column in df.columns:
                    df[column] = df[column].replace(replacements)
                    logger.info(f"Replaced values in column '{column}': {replacements}")
                else:
                    logger.warning(
                        f"Column '{column}' not found. Skipping replacements."
                    )

    def _remove_empty_rows(self, df):
        """Removes rows where all values are NaN."""
        if not df.empty:
            num_rows_before = len(df)
            df.dropna(how="all", inplace=True)
            num_rows_after = len(df)
            removed_count = num_rows_before - num_rows_after
            logger.info(f"Removed {removed_count} completely empty rows.")

    def _log_dataset_info(self, df):
        """Logs dataset shape, missing values per column, and missing rows."""
        if df is not None:
            num_rows, num_cols = df.shape
            missing_values_per_column = df.isna().sum()
            columns_with_missing_values = missing_values_per_column[
                missing_values_per_column > 0
            ]

            logger.info(f"Dataset Shape: {num_rows} rows, {num_cols} columns")
            if not columns_with_missing_values.empty:
                logger.info(
                    "Columns with Missing Values:\n" + str(columns_with_missing_values)
                )
            else:
                logger.info("No columns with missing values.")

    def _check_missing_columns(self, df, columns):
        """Checks if any columns are missing before performing operations."""
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"Some specified columns were not found in the dataset: {missing_cols}"
            )
        return [col for col in columns if col in df.columns]


def main():
    try:
        config_path, csv_path, result_path = check_args_paths(
            description="Clean a dataset.",
            config_help="Path to the JSON configuration file.",
            csv_help="Path to the input dataset CSV file.",
            result_help="Path to save the final processed dataset CSV file.",
        )
    except FileNotFoundError as e:
        logger.error(e)
        print(e)
        return

    df = pd.read_csv(csv_path)

    cleaner = DataCleaner(config_json_file=config_path)
    cleaned_df = cleaner.clean_data(df)

    cleaned_df.to_csv(result_path, index=False)
    logger.info(f"Processed dataset saved to {result_path}")


if __name__ == "__main__":
    main()
