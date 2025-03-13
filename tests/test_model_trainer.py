"""
This module contains unit tests for the ModelTrainer class, which handles model training and saving.
Author: Marc Ennaji
Date: 2025-03-01
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from common.exceptions import ModelTrainingError, ConfigValidationError
from models.model_trainer import ModelTrainer

# =========================== FIXTURES =========================== #


@pytest.fixture
def sample_training_config_fixture():
    """Returns a sample training configuration."""
    return {
        "RandomForestClassifier": {
            "n_estimators": [100],
            "max_depth": [10],
            "grid_search": {"cv": 3, "n_jobs": -1},
        },
        "LogisticRegression": {
            "solver": ["lbfgs"],
            "max_iter": [200],
            "grid_search": {"cv": 5, "n_jobs": -1},
        },
    }


@pytest.fixture
def sample_data_fixture():
    """Returns sample feature and target datasets."""
    X = pd.DataFrame({"feature1": range(10), "feature2": range(10, 20)})
    y = pd.Series([0, 1] * 5)  # Binary classification labels
    return X, y


@pytest.fixture
def model_trainer_fixture(sample_training_config_fixture):
    """Returns a ModelTrainer instance with a sample config."""
    return ModelTrainer(training_config=sample_training_config_fixture)


# =========================== TEST INITIALIZATION =========================== #


def test_model_trainer_initialization(sample_training_config_fixture):
    """Test successful initialization of ModelTrainer."""
    trainer = ModelTrainer(training_config=sample_training_config_fixture)
    assert isinstance(trainer.training_config, dict)
    assert len(trainer.models) == 2  # Expecting two models


def test_model_trainer_invalid_config():
    """Test invalid config type raises ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="Invalid training configuration"):
        ModelTrainer(training_config="invalid_config")  # Not a dict


def test_model_trainer_empty_config():
    """Test empty config raises ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="No valid models found"):
        ModelTrainer(training_config={})


# =========================== TEST INPUT VALIDATION =========================== #


def test_validate_inputs_valid(model_trainer_fixture, sample_data_fixture):
    """Test that valid inputs pass validation."""
    X_train, y_train = sample_data_fixture
    model_trainer_fixture._validate_inputs(
        X_train, y_train
    )  # Should not raise an error


@pytest.mark.parametrize(
    "invalid_X, invalid_y",
    [
        (None, pd.Series([0, 1] * 5)),  # X_train is None
        (pd.DataFrame(), pd.Series([0, 1] * 5)),  # Empty X_train
        (pd.DataFrame({"feature1": range(10)}), None),  # y_train is None
        (pd.DataFrame({"feature1": range(10)}), pd.Series()),  # Empty y_train
    ],
)
def test_validate_inputs_invalid(model_trainer_fixture, invalid_X, invalid_y):
    """Test that invalid inputs raise ModelTrainingError."""
    with pytest.raises(ModelTrainingError):
        model_trainer_fixture._validate_inputs(invalid_X, invalid_y)


# =========================== TEST MODEL TRAINING =========================== #


@patch("models.model_trainer.GridSearchCV")
def test_grid_search_success(
    mock_grid_search, model_trainer_fixture, sample_data_fixture
):
    """Test grid search successfully finds the best model."""
    X_train, y_train = sample_data_fixture
    mock_best_model = MagicMock()

    mock_grid_search_instance = MagicMock()
    mock_grid_search_instance.best_estimator_ = mock_best_model
    mock_grid_search.return_value = mock_grid_search_instance

    best_model = model_trainer_fixture._perform_grid_search(
        RandomForestClassifier(), {"n_estimators": [100]}, {"cv": 3}, X_train, y_train
    )

    assert best_model == mock_best_model
    mock_grid_search_instance.fit.assert_called_once()


@patch("models.model_trainer.GridSearchCV")
def test_grid_search_failure(
    mock_grid_search, model_trainer_fixture, sample_data_fixture
):
    """Test that grid search failure raises ModelTrainingError."""
    X_train, y_train = sample_data_fixture
    mock_grid_search.side_effect = Exception("Grid search error")

    with pytest.raises(ModelTrainingError, match="Error during grid search"):
        model_trainer_fixture._perform_grid_search(
            RandomForestClassifier(),
            {"n_estimators": [100]},
            {"cv": 3},
            X_train,
            y_train,
        )


@patch.object(RandomForestClassifier, "fit")
@patch.object(LogisticRegression, "fit")
def test_train_models(
    mock_rf_fit, mock_lr_fit, model_trainer_fixture, sample_data_fixture
):
    """Test training all models successfully."""
    X_train, y_train = sample_data_fixture
    trained_models = model_trainer_fixture.train(X_train, y_train)

    assert "RandomForestClassifier" in trained_models
    assert "LogisticRegression" in trained_models
    mock_rf_fit.assert_called_once()
    mock_lr_fit.assert_called_once()


@patch.object(RandomForestClassifier, "fit", side_effect=Exception("RF Training Error"))
def test_train_model_failure(mock_rf_fit, model_trainer_fixture, sample_data_fixture):
    """Test that a training failure raises ModelTrainingError."""
    X_train, y_train = sample_data_fixture

    with pytest.raises(
        ModelTrainingError, match="Error training RandomForestClassifier"
    ):
        model_trainer_fixture.train(X_train, y_train)


# =========================== TEST MODEL SAVING =========================== #


def test_save_models(tmp_path, model_trainer_fixture):
    """Test saving trained models."""
    model_dir = tmp_path / "models"
    trained_models = {
        "RandomForestClassifier": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(),
    }

    model_trainer_fixture.save_models(trained_models, str(model_dir))

    assert os.path.exists(model_dir / "RandomForestClassifier.pkl")
    assert os.path.exists(model_dir / "LogisticRegression.pkl")


def test_save_models_failure(tmp_path, model_trainer_fixture):
    """Test failure during model saving."""
    model_dir = tmp_path / "models"
    trained_models = {"RandomForestClassifier": None}

    with pytest.raises(
        ModelTrainingError,
        match="Model 'RandomForestClassifier' should be an instance of RandomForestClassifier.",
    ):
        model_trainer_fixture.save_models(trained_models, str(model_dir))
