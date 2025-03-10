from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from common.exceptions import ModelTrainingError, ConfigValidationError
from logger_config import logger

# Dictionary mapping model names to their corresponding classes
MODEL_MAPPING = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
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
            list: A list of tuples (model_instance, hyperparameter_grid).

        Raises:
            ConfigValidationError: If no valid models are found.
        """
        models = []
        for model_name, model_class in MODEL_MAPPING.items():
            if model_name in self.training_config:
                hyperparameters = self.training_config.get(model_name, {})
                models.append(
                    (model_class(random_state=self.random_state), hyperparameters)
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
        trained_models = {}

        for model, param_grid in self.models:
            model_name = type(model).__name__

            try:
                if param_grid:
                    logger.info(f"Performing Grid Search for {model_name}...")
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        **self.training_config.get("grid_search", {}),
                    )
                    grid_search.fit(X_train, y_train)
                    trained_models[model_name] = grid_search.best_estimator_
                else:
                    logger.info(f"Training {model_name} with default parameters...")
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model

                logger.info(f"Successfully trained {model_name}.")
            except Exception as e:
                raise ModelTrainingError(
                    f"Error training {model_name}: {str(e)}"
                ) from e

        return trained_models
