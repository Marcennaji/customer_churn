import pandas as pd
from sklearn.model_selection import train_test_split

from common.exceptions import DataSplittingError, ConfigValidationError


class DatasetSplitter:
    """Handles train-test data splitting based on JSON configuration profiles."""

    def __init__(
            self,
            df: pd.DataFrame,
            config: dict,
            profile: str = "default"):
        """
        Initializes DatasetSplitter using a chosen profile.

        Args:
            df (pd.DataFrame): The dataset.
            config (dict): Configuration dictionary containing multiple profiles.
            profile (str): Name of the profile to use. Default is "default".

        Raises:
            ConfigValidationError: If the profile is missing or invalid.
        """
        if not isinstance(config, dict):
            raise ConfigValidationError(
                "Invalid configuration. Expected a dictionary.")

        if df is None or df.empty:
            raise DataSplittingError(
                "Input DataFrame is empty or None. Cannot perform splitting."
            )

        self.df = df
        self.profile = profile
        self.config = config

        self._apply_profile()

    def _apply_profile(self):
        """Applies the selected profile's settings."""
        if self.profile not in self.config:
            raise ConfigValidationError(
                f"Profile '{self.profile}' not found in config file."
            )

        profile_config = self.config[self.profile]

        try:
            self.feature_columns = profile_config["feature_columns"]
            self.target_column = profile_config["target_column"]
            self.target_type = profile_config.get("target_type", None)
            self.test_size = profile_config.get("test_size", 0.3)
            self.random_state = profile_config.get("random_state", 42)

            if not self.feature_columns or not isinstance(
                    self.feature_columns, list):
                raise ConfigValidationError(
                    f"Invalid feature_columns list in profile '{self.profile}'."
                )

            if not isinstance(
                    self.target_column,
                    str) or not self.target_column:
                raise ConfigValidationError(
                    f"Invalid target_column in profile '{self.profile}'."
                )

        except KeyError as e:
            raise ConfigValidationError(
                f"Missing required key in profile '{self.profile}': {str(e)}"
            ) from e

    def split(self):
        """
        Extracts X and y, then splits the dataset into training and test sets.

        Returns:
            tuple: X_train, X_test, y_train, y_test

        Raises:
            DataSplittingError: If splitting fails or required columns are missing.
        """
        try:
            # Ensure all required columns exist in the dataset
            missing_features = [
                col for col in self.feature_columns if col not in self.df.columns]
            if missing_features:
                raise DataSplittingError(
                    f"Missing feature columns in dataset: {missing_features}"
                )

            if self.target_column not in self.df.columns:
                raise DataSplittingError(
                    f"Target column '{self.target_column}' not found in dataset."
                )

            X = self.df[self.feature_columns]
            y = self.df[self.target_column]

            # Convert y to the specified target type
            if self.target_type:
                try:
                    y = y.astype(self.target_type)
                except ValueError:
                    raise DataSplittingError(
                        f"Could not convert target column '{self.target_column}' to type {self.target_type}."
                    )

            return train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        except Exception as e:
            raise DataSplittingError(
                f"Error during train-test splitting: {str(e)}"
            ) from e
