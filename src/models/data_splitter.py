"""
This module handles train-test data splitting based on JSON configuration profiles for the customer churn project.
Author: Marc Ennaji
Date: 2025-03-01
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from common.exceptions import DataSplittingError, ConfigValidationError
from logger_config import get_logger


class DatasetSplitter:
    """Handles train-test data splitting based on JSON configuration profiles."""

    def __init__(self, df: pd.DataFrame, config: dict, profile: str = "default"):
        """
        Initializes DatasetSplitter using a chosen profile.

        Args:
            df (pd.DataFrame): The dataset.
            config (dict): Configuration dictionary containing multiple profiles.
            profile (str): Name of the profile to use. Default is "default".

        Raises:
            ConfigValidationError: If the profile is missing or invalid.
        """
        get_logger().info("Initializing DatasetSplitter")
        if not isinstance(config, dict):
            message = "Invalid configuration. Expected a dictionary."
            get_logger().error(message)
            raise ConfigValidationError(message)

        if df is None or df.empty:
            message = "Input DataFrame is empty or None. Cannot perform splitting."
            get_logger().error(message)
            raise DataSplittingError(message)

        self.df = df
        self.config = config

        self.profile_config = self.apply_profile(profile)

    def apply_profile(self, profile: str):
        """Applies the selected profile's settings and returns the configuration."""
        get_logger().info("Applying profile: %s", profile)
        if profile not in self.config:
            message = "Profile '%s' not found in config file." % profile
            get_logger().error(message)
            raise ConfigValidationError(message)
        self.profile = profile
        profile_config = self.config[self.profile]

        try:
            feature_columns = profile_config["feature_columns"]
            target_column = profile_config["target_column"]
            target_type = profile_config.get("target_type", None)
            test_size = profile_config.get("test_size", 0.3)
            random_state = profile_config.get("random_state", 42)

            if not feature_columns or not isinstance(feature_columns, list):
                message = "Invalid feature_columns list in profile '%s'." % self.profile
                get_logger().error(message)
                raise ConfigValidationError(message)

            if not isinstance(target_column, str) or not target_column:
                message = "Invalid target_column in profile '%s'." % self.profile
                get_logger().error(message)
                raise ConfigValidationError(message)

            return {
                "feature_columns": feature_columns,
                "target_column": target_column,
                "target_type": target_type,
                "test_size": test_size,
                "random_state": random_state,
            }

        except KeyError as e:
            message = "Missing required key in profile '%s': %s" % (
                self.profile,
                str(e),
            )
            get_logger().error(message)
            raise ConfigValidationError(message) from e

    def split(self):
        """
        Extracts X and y, then splits the dataset into training and test sets.

        Returns:
            tuple: X_train, X_test, y_train, y_test

        Raises:
            DataSplittingError: If splitting fails or required columns are missing.
        """
        get_logger().info("Splitting dataset")
        try:
            # Ensure all required columns exist in the dataset
            missing_features = [
                col
                for col in self.profile_config["feature_columns"]
                if col not in self.df.columns
            ]
            if missing_features:
                message = "Missing feature columns in dataset: %s" % missing_features
                get_logger().error(message)
                raise DataSplittingError(message)

            if self.profile_config["target_column"] not in self.df.columns:
                message = (
                    "Target column '%s' not found in dataset."
                    % self.profile_config["target_column"]
                )
                get_logger().error(message)
                raise DataSplittingError(message)

            X = self.df[self.profile_config["feature_columns"]]
            y = self.df[self.profile_config["target_column"]]

            # Convert y to the specified target type
            if self.profile_config["target_type"]:
                try:
                    y = y.astype(self.profile_config["target_type"])
                except ValueError as exc:
                    message = "Could not convert target column '%s' to type %s." % (
                        self.profile_config["target_column"],
                        self.profile_config["target_type"],
                    )
                    get_logger().error(message)
                    raise DataSplittingError(message) from exc

            return train_test_split(
                X,
                y,
                test_size=self.profile_config["test_size"],
                random_state=self.profile_config["random_state"],
            )

        except Exception as e:
            message = "Error during train-test splitting: %s" % str(e)
            get_logger().error(message)
            raise DataSplittingError(message) from e
