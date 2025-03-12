"""
This module defines custom exceptions for the customer churn project.
Author: Marc Ennaji
Date: 2023-10-10
"""


class MLPipelineError(Exception):
    """Base class for all exceptions in the ML pipeline."""

    def __init__(self, message="An error occurred in the ML pipeline."):
        super().__init__(message)


# ================= DATA EXCEPTIONS ================= #
class DataError(MLPipelineError):
    """Base class for data-related exceptions."""


class DataLoadingError(DataError):
    """Raised when there is an error loading data."""


class DataValidationError(DataError):
    """Raised when data validation fails."""


class DataPreprocessingError(DataError):
    """Raised when data preprocessing encounters an issue."""


class DataEncodingError(DataPreprocessingError):
    """Raised when data encoding fails."""


class DataSplittingError(DataError):
    """Raised when data splitting fails."""


# ================= FEATURE ENGINEERING ================= #
class FeatureEngineeringError(DataError):
    """Raised when feature engineering fails."""


# ================= MODEL EXCEPTIONS ================= #
class ModelError(MLPipelineError):
    """Base class for model-related exceptions."""


class ModelTrainingError(ModelError):
    """Raised when model training fails."""


class ModelEvaluationError(ModelError):
    """Raised when model evaluation fails."""


class ModelSaveError(ModelError):
    """Raised when saving a model fails."""


class ModelLoadError(ModelError):
    """Raised when loading a model fails."""


# ================= CONFIGURATION EXCEPTIONS ================= #
class ConfigError(MLPipelineError):
    """Base class for configuration-related exceptions."""


class ConfigLoadingError(ConfigError):
    """Raised when there is an error loading configuration."""


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""


# ================= LOGGER EXCEPTIONS ================= #
class LoggerConfigurationError(MLPipelineError):
    """Raised when configuring the logger fails."""
