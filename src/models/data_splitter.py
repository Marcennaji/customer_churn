import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Handles train-test data splitting based on JSON configuration profiles."""

    def __init__(self, df: pd.DataFrame, config, profile: str = "default"):
        """
        Initializes DatasetSplitter using a specific experiment profile.

        Args:
            df (pd.DataFrame): The dataset.
            config_path (str): Path to JSON config file containing multiple profiles.
            profile (str): Name of the profile to use. Default is "default".
        """
        self.df = df
        self.profile = profile
        self.config = config
        self._apply_profile()

    def _apply_profile(self):
        """Applies the selected profile's settings."""
        if self.profile not in self.config:
            raise ValueError(f"Profile '{self.profile}' not found in config file.")

        profile_config = self.config[self.profile]
        self.feature_columns = profile_config["feature_columns"]
        self.target_column = profile_config["target_column"]
        self.test_size = profile_config.get("test_size", 0.3)
        self.random_state = profile_config.get("random_state", 42)

    def split(self):
        """Extracts X and y, then splits the dataset into training and test sets."""
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
