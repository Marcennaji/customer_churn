import pytest
import pandas as pd
import os
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.models.model_trainer import ModelTrainer
from common.exceptions import ModelTrainingError, ConfigValidationError


@pytest.fixture
def training_config():
    return {
        "random_forest": {
            "n_estimators": [100],
            "max_depth": [10],
            "grid_search": {
                "cv": 2,
                "scoring": "accuracy",
                "n_jobs": -1,
                "verbose": 1,
                "error_score": "raise",
            },
        },
        "logistic_regression": {
            "penalty": ["l2"],
            "C": [0.1, 1],
            "solver": ["lbfgs"],
            "max_iter": [100],
        },
    }


@pytest.fixture
def X_train():
    return pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]})


@pytest.fixture
def y_train():
    return pd.Series([0, 1, 0, 1])


def test_initialization(training_config):
    trainer = ModelTrainer(training_config=training_config)
    assert trainer.training_config == training_config
    assert trainer.random_state == 42
    assert len(trainer.models) == 2


def test_invalid_training_config():
    with pytest.raises(ConfigValidationError):
        ModelTrainer(training_config="invalid_config")


def test_validate_inputs(training_config, X_train, y_train):
    trainer = ModelTrainer(training_config=training_config)
    trainer._validate_inputs(X_train, y_train)

    with pytest.raises(ModelTrainingError):
        trainer._validate_inputs(pd.DataFrame(), y_train)

    with pytest.raises(ModelTrainingError):
        trainer._validate_inputs(X_train, pd.Series())


def test_initialize_models(training_config):
    trainer = ModelTrainer(training_config=training_config)
    models = trainer._initialize_models()
    assert len(models) == 2
    assert isinstance(models[0][0], RandomForestClassifier)
    assert isinstance(models[1][0], LogisticRegression)


@patch("src.models.model_trainer.GridSearchCV")
def test_perform_grid_search(mock_grid_search, X_train, y_train, training_config):
    trainer = ModelTrainer(training_config=training_config)
    model = RandomForestClassifier(random_state=42)
    param_grid = training_config["random_forest"]
    grid_search_config = param_grid.pop("grid_search")

    # Create a mock instance of GridSearchCV
    mock_grid_search_instance = mock_grid_search.return_value
    # Set the best_estimator_ attribute on the mock instance
    mock_grid_search_instance.best_estimator_ = model
    # Ensure the fit method returns the mock instance itself
    mock_grid_search_instance.fit.return_value = mock_grid_search_instance

    best_model = trainer._perform_grid_search(
        model, param_grid, grid_search_config, X_train, y_train
    )
    assert best_model == model


def test_train(X_train, y_train, training_config):
    trainer = ModelTrainer(training_config=training_config)
    trained_models = trainer.train(X_train, y_train)
    assert "RandomForestClassifier" in trained_models
    assert "LogisticRegression" in trained_models


@patch("src.models.model_trainer.joblib.dump")
def test_save_models(mock_joblib_dump, training_config):
    trainer = ModelTrainer(training_config=training_config)
    trained_models = {
        "RandomForestClassifier": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(),
    }
    directory = "models"
    trainer.save_models(trained_models, directory)
    assert mock_joblib_dump.call_count == 2
    mock_joblib_dump.assert_any_call(
        trained_models["RandomForestClassifier"],
        os.path.join(directory, "RandomForestClassifier.joblib"),
    )
    mock_joblib_dump.assert_any_call(
        trained_models["LogisticRegression"],
        os.path.join(directory, "LogisticRegression.joblib"),
    )
