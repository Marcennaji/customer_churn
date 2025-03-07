class MLPipelineError(Exception):
    """Base class for all exceptions in the ML pipeline."""

    pass


# Data-related exceptions
class DataError(MLPipelineError):
    pass


class DataLoadingError(DataError):
    pass


class DataValidationError(DataError):
    pass


class DataPreprocessingError(DataError):
    pass


class DataEncodingError(DataPreprocessingError):
    pass


# Feature engineering exceptions
class FeatureEngineeringError(MLPipelineError):
    pass


# Model-related exceptions
class ModelError(MLPipelineError):
    pass


class ModelTrainingError(ModelError):
    pass


class DataSplittingError(ModelTrainingError):
    pass


class ModelEvaluationError(ModelError):
    pass


class ModelSaveError(ModelError):
    pass


class ModelLoadError(ModelError):
    pass


# Configuration exceptions
class ConfigError(MLPipelineError):
    pass


class ConfigLoadingError(ConfigError):
    pass


class ConfigValidationError(ConfigError):
    pass


# Utility exceptions
class LoggerConfigurationError(MLPipelineError):
    pass
