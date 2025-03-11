import argparse
import os
import json
from logger_config import logger
from common.exceptions import ConfigLoadingError


class ConfigManager:
    """Handles argument parsing and configuration file loading with logging."""

    def __init__(
        self,
        description,
        csv_help="Path to the input dataset CSV file.",
        data_dir_help="Directory to save the processed dataset CSV files.",
        models_dir_help="Directory to save the trained models.",
        preprocessing_help="Path to the preprocessing config JSON file.",
        splitting_help="Path to the data splitting config JSON file.",
        training_help="Path to the model training config JSON file.",
        eval_only_help="Enable evaluation-only mode (skip training).",
        default_preprocessing="config/preprocessing_config.json",
        default_splitting="config/data_splitting_profiles.json",
        default_training="config/training_config.json",
        default_data_dir="data",
        default_models_dir="models",
    ):
        """
        Initializes the ConfigManager and parses command-line arguments.
        """
        self.description = description
        self.default_preprocessing = default_preprocessing
        self.default_splitting = default_splitting
        self.default_training = default_training
        self.default_data_dir = default_data_dir
        self.default_models_dir = default_models_dir

        self.args = self._parse_arguments(
            csv_help,
            data_dir_help,
            models_dir_help,
            preprocessing_help,
            splitting_help,
            training_help,
            eval_only_help,
        )

        self._validate_paths()
        self.configs = self._load_configs()

    def _parse_arguments(
        self,
        csv_help,
        data_dir_help,
        models_dir_help,
        preprocessing_help,
        splitting_help,
        training_help,
        eval_only_help,
    ):
        """Parses command-line arguments."""
        parser = argparse.ArgumentParser(description=self.description)

        # Required CSV file path
        parser.add_argument("--csv", type=str, required=True, help=csv_help)

        # Directories for data and models
        parser.add_argument(
            "--data-dir",
            type=str,
            default=self.default_data_dir,
            help=data_dir_help)
        parser.add_argument(
            "--models-dir",
            type=str,
            default=self.default_models_dir,
            help=models_dir_help,
        )

        # Config file paths (optional, with defaults)
        parser.add_argument(
            "--preprocessing-config",
            type=str,
            default=self.default_preprocessing,
            help=preprocessing_help,
        )
        parser.add_argument(
            "--splitting-config",
            type=str,
            default=self.default_splitting,
            help=splitting_help,
        )
        parser.add_argument(
            "--training-config",
            type=str,
            default=self.default_training,
            help=training_help,
        )

        # Evaluation-only flag (skip training)
        parser.add_argument(
            "--eval-only",
            action="store_true",
            help=eval_only_help)

        return parser.parse_args()

    def _validate_paths(self):
        """Validates paths for CSV and config files."""
        if not os.path.isfile(self.args.csv):
            logger.error(f"CSV file not found: {self.args.csv}")
            raise ConfigLoadingError(f"CSV file not found: {self.args.csv}")

        if not os.path.isdir(self.args.data_dir):
            logger.error(
                f"Data directory does not exist: {self.args.data_dir}")
            raise ConfigLoadingError(
                f"Data directory does not exist: {self.args.data_dir}"
            )

        if not os.path.isdir(self.args.models_dir):
            logger.error(
                f"Models directory does not exist: {self.args.models_dir}")
            raise ConfigLoadingError(
                f"Models directory does not exist: {self.args.models_dir}"
            )

        # Check if each config file exists, log warnings if missing
        for config_name, config_path in {
            "Preprocessing config": self.args.preprocessing_config,
            "Splitting config": self.args.splitting_config,
            "Training config": self.args.training_config,
        }.items():
            if not os.path.isfile(config_path):
                logger.warning(
                    f"{config_name} not found at '{config_path}', using default values."
                )
            else:
                logger.info(f"{config_name} loaded from '{config_path}'.")

    def _load_configs(self):
        """Loads JSON configurations from files."""
        configs = {}
        for config_name, config_path in {
            "preprocessing": self.args.preprocessing_config,
            "splitting": self.args.splitting_config,
            "training": self.args.training_config,
        }.items():
            try:
                with open(config_path, "r") as file:
                    configs[config_name] = json.load(file)
                    logger.info(
                        f"{config_name.capitalize()} config successfully loaded from '{config_path}'."
                    )
            except ConfigLoadingError:
                configs[config_name] = {}
                logger.warning(
                    f"{config_name.capitalize()} config not found, using default empty config."
                )

        return configs

    def get_config(self, config_type):
        """Retrieves a specific configuration dictionary (preprocessing, splitting, training)."""
        return self.configs.get(config_type, {})

    def get_csv_path(self):
        """Returns the CSV file path."""
        return self.args.csv

    def get_data_dir(self):
        """Returns the data directory path."""
        return self.args.data_dir

    def get_models_dir(self):
        """Returns the models directory path."""
        return self.args.models_dir

    def is_eval_only(self):
        """Returns True if evaluation-only mode is enabled."""
        return self.args.eval_only
