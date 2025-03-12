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
        logger.info("CSV path: %s", self.config["csv"])

        self.config["data_dir"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["data_dir"])
        )
        logger.info("Data directory path: %s", self.config["data_dir"])

        self.config["models_dir"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["models_dir"])
        )
        logger.info("Models directory path: %s", self.config["models_dir"])

        self.config["results_dir"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["results_dir"])
        )
        logger.info("Results directory path: %s", self.config["results_dir"])

        self.config["preprocessing"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["preprocessing"])
        )
        logger.info("Preprocessing config path: %s", self.config["preprocessing"])

        self.config["splitting"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["splitting"])
        )
        logger.info("Splitting config path: %s", self.config["splitting"])

        self.config["training"] = os.path.abspath(
            os.path.join(self.root_directory, self.config["training"])
        )
        logger.info("Training config path: %s", self.config["training"])

    def _validate_paths(self):
        """Validates paths for CSV and config files."""
        self._validate_file_path(self.config["csv"], "CSV file")
        self._validate_directory_path(self.config["data_dir"], "Data directory")
        self._validate_directory_path(self.config["models_dir"], "Models directory")
        self._validate_directory_path(self.config["results_dir"], "Results directory")
        self._validate_file_path(
            self.config["preprocessing"], "Preprocessing config file"
        )
        self._validate_file_path(self.config["splitting"], "Splitting config file")
        self._validate_file_path(self.config["training"], "Training config file")

    def _validate_file_path(self, path, description):
        """Validates that a file path exists."""
        if not os.path.isfile(path):
            logger.error("%s not found: %s", description, path)
            raise ConfigLoadingError(f"{description} not found: {path}")

    def _validate_directory_path(self, path, description):
        """Validates that a directory path exists."""
        if not os.path.isdir(path):
            logger.error("%s does not exist: %s", description, path)
            raise ConfigLoadingError(f"{description} does not exist: {path}")

    def _load_nested_json_data(self):
        """Loads JSON data for preprocessing, splitting, and training configurations."""
        self.config["preprocessing"] = self._load_json_data(
            self.config["preprocessing"]
        )
        logger.info("Preprocessing config data loaded.")

        self.config["splitting"] = self._load_json_data(self.config["splitting"])
        logger.info("Splitting config data loaded.")

        self.config["training"] = self._load_json_data(self.config["training"])
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

    def get_results_dir(self):
        """Returns the results directory path."""
        return self.config["results_dir"]

    def is_eval_only(self):
        """Returns True if evaluation-only mode is enabled."""
        return self.config.get("eval_only", False)
