import os
import json
from logger_config import logger
from common.exceptions import ConfigLoadingError


class ConfigManager:
    """Handles configuration loading from a JSON file with logging."""

    def __init__(self, config_file_path):
        """
        Initializes the ConfigManager and loads configuration from a JSON file.
        """
        self.config_file_path = config_file_path
        self.config = self._load_config_file()
        self.root_directory = self.config.get("root_directory", "")
        self._convert_to_paths()
        self._validate_paths()
        self._load_nested_json_data()

    def _load_config_file(self):
        """Loads the configuration from the JSON file."""
        if not os.path.isfile(self.config_file_path):
            logger.error("Config file not found: %s", self.config_file_path)
            raise ConfigLoadingError(f"Config file not found: {self.config_file_path}")

        with open(self.config_file_path, "r") as file:
            config = json.load(file)
            logger.info("Config file loaded from '%s'.", self.config_file_path)
            return config

    def _load_json_data(self, file_path):
        """Loads JSON data from a file."""
        if not os.path.isfile(file_path):
            logger.error("JSON file not found: %s", file_path)
            raise ConfigLoadingError(f"JSON file not found: {file_path}")

        with open(file_path, "r") as file:
            data = json.load(file)
            logger.info("JSON data loaded from '%s'.", file_path)
            return data

    def _convert_to_paths(self):
        """Converts relative paths to absolute paths based on the root directory."""
        self.config["csv"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["csv"])
        )
        self.config["data_dir"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["data_dir"])
        )
        self.config["models_dir"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["models_dir"])
        )
        self.config["preprocessing"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["preprocessing"])
        )
        self.config["splitting"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["splitting"])
        )
        self.config["training"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["training"])
        )

        # Log the converted paths for debugging
        logger.info("CSV path: %s", self.config["csv"])
        logger.info("Data directory path: %s", self.config["data_dir"])
        logger.info("Models directory path: %s", self.config["models_dir"])
        logger.info("Preprocessing config path: %s", self.config["preprocessing"])
        logger.info("Splitting config path: %s", self.config["splitting"])
        logger.info("Training config path: %s", self.config["training"])

    def _validate_paths(self):
        """Validates paths for CSV and config files."""
        if not os.path.isfile(self.config["csv"]):
            logger.error("CSV file not found: %s", self.config["csv"])
            raise ConfigLoadingError(f"CSV file not found: {self.config['csv']}")

        if not os.path.isdir(self.config["data_dir"]):
            logger.error("Data directory does not exist: %s", self.config["data_dir"])
            raise ConfigLoadingError(
                f"Data directory does not exist: {self.config['data_dir']}"
            )

        if not os.path.isdir(self.config["models_dir"]):
            logger.error(
                "Models directory does not exist: %s", self.config["models_dir"]
            )
            raise ConfigLoadingError(
                f"Models directory does not exist: {self.config['models_dir']}"
            )

        if not os.path.isfile(self.config["preprocessing"]):
            logger.error(
                "Preprocessing config file not found: %s", self.config["preprocessing"]
            )
            raise ConfigLoadingError(
                f"Preprocessing config file not found: {self.config['preprocessing']}"
            )

        if not os.path.isfile(self.config["splitting"]):
            logger.error(
                "Splitting config file not found: %s", self.config["splitting"]
            )
            raise ConfigLoadingError(
                f"Splitting config file not found: {self.config['splitting']}"
            )

        if not os.path.isfile(self.config["training"]):
            logger.error("Training config file not found: %s", self.config["training"])
            raise ConfigLoadingError(
                f"Training config file not found: {self.config['training']}"
            )

    def _load_nested_json_data(self):
        """Loads JSON data for preprocessing, splitting, and training configurations."""
        self.config["preprocessing"] = self._load_json_data(
            self.config["preprocessing"]
        )
        self.config["splitting"] = self._load_json_data(self.config["splitting"])
        self.config["training"] = self._load_json_data(self.config["training"])

        # Log the loaded JSON data for debugging
        logger.info("Preprocessing config data loaded.")
        logger.info("Splitting config data loaded.")
        logger.info("Training config data loaded.")

    def get_config(self, config_type):
        """Retrieves a specific configuration dictionary (preprocessing, splitting, training)."""
        return self.config.get(config_type, {})

    def get_csv_path(self):
        """Returns the CSV file path."""
        return self.config["csv"]

    def get_data_dir(self):
        """Returns the data directory path."""
        return self.config["data_dir"]

    def get_models_dir(self):
        """Returns the models directory path."""
        return self.config["models_dir"]

    def is_eval_only(self):
        """Returns True if evaluation-only mode is enabled."""
        return self.config.get("eval_only", False)
