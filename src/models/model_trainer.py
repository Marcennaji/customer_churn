"""
This module handles the training and hyperparameter tuning of models for the customer churn project.
Author: Marc Ennaji
Date: 2023-10-10
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from common.exceptions import ModelTrainingError, ConfigValidationError
from logger_config import logger
import pandas as pd
import os
import joblib


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
        if not isinstance(training_config, dict):
            raise ConfigValidationError(
                "Invalid training configuration. Expected a dictionary."
            )

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
                logger.warning(
                    f"Model '{model_name}' not found in training config. Skipping."
                )

        if not models:
            raise ConfigValidationError(
                "No valid models found in the training configuration."
            )

        return models

    def _validate_inputs(self, X_train, y_train):
        """
        Validates the input training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Raises:
            ModelTrainingError: If the inputs are invalid.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ModelTrainingError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise ModelTrainingError("y_train must be a pandas Series.")
        if X_train.empty:
            raise ModelTrainingError("X_train is empty.")
        if y_train.empty:
            raise ModelTrainingError("y_train is empty.")

    def _perform_grid_search(
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
        try:

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                **grid_search_config,
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        except Exception as e:
            raise ModelTrainingError(
                f"Error during grid search for {type(model).__name__}: {str(e)}"
            ) from e

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
        # Validate inputs
        self._validate_inputs(X_train, y_train)

        trained_models = {}

        for model, param_grid, grid_search_config in self.models:
            model_name = type(model).__name__

            try:
                logger.info(
                    f"Training {model_name} with hyperparameters: {model.get_params()}"
                )
                if param_grid and grid_search_config:
                    logger.info(f"Performing Grid Search for {model_name}...")
                    logger.info(
                        f"Grid Search parameters for {model_name}: {param_grid}"
                    )
                    best_model = self._perform_grid_search(
                        model, param_grid, grid_search_config, X_train, y_train
                    )
                    trained_models[model_name] = best_model
                else:
                    logger.info(f"Training {model_name} without Grid Search...")
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model

                logger.info(f"Successfully trained {model_name}.")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                raise ModelTrainingError(
                    f"Error training {model_name}: {str(e)}"
                ) from e

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
        if not os.path.exists(directory):
            os.makedirs(directory)

        for model_name, model in trained_models.items():

            if model_name not in MODEL_MAPPING:
                raise ModelTrainingError(f"Model name '{model_name}' is not handled.")

            expected_model_class = MODEL_MAPPING[model_name]
            if not isinstance(model, expected_model_class):
                raise ModelTrainingError(
                    f"Model '{model_name}' should be an instance of {expected_model_class.__name__}."
                )

            model_path = os.path.join(directory, f"{model_name}.pkl")
            try:
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
                raise ModelTrainingError(f"Error saving {model_name}: {str(e)}") from e
