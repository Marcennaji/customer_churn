"""
This module handles the training and hyperparameter tuning of models for the customer churn project.
Author: Marc Ennaji
Date: 2025-03-01
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from common.exceptions import ModelTrainingError, ConfigValidationError
from logger_config import get_logger


# Dictionary mapping model names to their corresponding classes
MODEL_MAPPING = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
}


class ModelTrainer:
    """Handles training and hyperparameter tuning of models based on a given configuration."""

    def __init__(self, training_config: dict, random_state=42):
        """
        Initializes ModelTrainer.

        Args:
            training_config (dict): Dictionary containing model configurations.
            random_state (int): Random seed for model reproducibility.

        Raises:
            ConfigValidationError: If the configuration is invalid.
        """
        get_logger().info("Initializing ModelTrainer")
        if not isinstance(training_config, dict):
            message = "Invalid training configuration. Expected a dictionary."
            get_logger().error(message)
            raise ConfigValidationError(message)

        self.random_state = random_state
        self.training_config = training_config
        self.models = self._initialize_models()

    def _initialize_models(self):
        """
        Initializes models based on the given configuration.

        Returns:
            list: A list of tuples (model_instance, hyperparameter_grid, grid_search_config).

        Raises:
            ConfigValidationError: If no valid models are found.
        """
        get_logger().info("Initializing models")
        models = []
        for model_name, model_class in MODEL_MAPPING.items():
            if model_name in self.training_config:
                config = self.training_config.get(model_name, {})
                hyperparameters = {
                    k: v for k, v in config.items() if k != "grid_search"
                }
                grid_search_config = config.get("grid_search", {})
                models.append(
                    (
                        model_class(**hyperparameters, random_state=self.random_state),
                        hyperparameters,
                        grid_search_config,
                    )
                )
            else:
                get_logger().warning(
                    "Model '%s' not found in training config. Skipping.", model_name
                )

        if not models:
            message = "No valid models found in the training configuration."
            get_logger().error(message)
            raise ConfigValidationError(message)

        return models

    def validate_inputs(self, X_train, y_train):
        """
        Validates the input training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Raises:
            ModelTrainingError: If the inputs are invalid.
        """
        get_logger().info("Validating inputs")
        if not isinstance(X_train, pd.DataFrame):
            message = "X_train must be a pandas DataFrame."
            get_logger().error(message)
            raise ModelTrainingError(message)
        if not isinstance(y_train, pd.Series):
            message = "y_train must be a pandas Series."
            get_logger().error(message)
            raise ModelTrainingError(message)
        if X_train.empty:
            message = "X_train is empty."
            get_logger().error(message)
            raise ModelTrainingError(message)
        if y_train.empty:
            message = "y_train is empty."
            get_logger().error(message)
            raise ModelTrainingError(message)

    def perform_grid_search(
        self, model, param_grid, grid_search_config, X_train, y_train
    ):
        """
        Performs grid search for hyperparameter tuning.

        Args:
            model: The model instance.
            param_grid: The hyperparameter grid.
            grid_search_config: The grid search configuration.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            The best estimator found by grid search.

        Raises:
            ModelTrainingError: If grid search fails.
        """
        get_logger().info("Performing grid search")
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                **grid_search_config,
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        except Exception as e:
            message = "Error during grid search for %s: %s" % (
                type(model).__name__,
                str(e),
            )
            get_logger().error(message)
            raise ModelTrainingError(message) from e

    def train(self, X_train, y_train):
        """
        Trains all configured models and returns the trained instances.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            dict: Dictionary of trained models.

        Raises:
            ModelTrainingError: If training fails.
        """
        get_logger().info("Training models")
        # Validate inputs
        self.validate_inputs(X_train, y_train)

        trained_models = {}

        for model, param_grid, grid_search_config in self.models:
            model_name = type(model).__name__

            try:
                get_logger().info(
                    "Training %s with hyperparameters: %s",
                    model_name,
                    model.get_params(),
                )
                if param_grid and grid_search_config:
                    get_logger().info("Performing Grid Search for %s...", model_name)
                    get_logger().info(
                        "Grid Search parameters for %s: %s", model_name, param_grid
                    )
                    best_model = self.perform_grid_search(
                        model, param_grid, grid_search_config, X_train, y_train
                    )
                    trained_models[model_name] = best_model
                else:
                    get_logger().info("Training %s without Grid Search...", model_name)
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model

                get_logger().info("Successfully trained %s.", model_name)
            except Exception as e:
                message = "Error training %s: %s" % (model_name, str(e))
                get_logger().error(message)
                raise ModelTrainingError(message) from e

        return trained_models

    def save_models(self, trained_models, directory):
        """
        Saves all trained models to the specified directory.

        Args:
            trained_models (dict): Dictionary of trained models.
            directory (str): Directory to save the models.

        Raises:
            ModelTrainingError: If saving the models fails.
        """
        get_logger().info("Saving models")
        if not os.path.exists(directory):
            os.makedirs(directory)

        for model_name, model in trained_models.items():
            if model_name not in MODEL_MAPPING:
                message = "Model name '%s' is not handled." % model_name
                get_logger().error(message)
                raise ModelTrainingError(message)

            expected_model_class = MODEL_MAPPING[model_name]
            if not isinstance(model, expected_model_class):
                message = "Model '%s' should be an instance of %s." % (
                    model_name,
                    expected_model_class.__name__,
                )
                get_logger().error(message)
                raise ModelTrainingError(message)

            model_path = os.path.join(directory, f"{model_name}.pkl")
            try:
                joblib.dump(model, model_path)
                get_logger().info("Saved %s to %s", model_name, model_path)
            except Exception as e:
                message = "Error saving %s: %s" % (model_name, str(e))
                get_logger().error(message)
                raise ModelTrainingError(message) from e
