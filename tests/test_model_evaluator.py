"""
This module contains unit tests for the ModelEvaluator class, which handles model evaluation, visualization, and reporting.
Author: Marc Ennaji
Date: 2025-03-01

Disabled the W0621 pylint warning, as it triggers a false positive when using fixtures (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint) (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint)

"""

# pylint: disable=W0621

import os
from unittest.mock import patch
import json
import joblib
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from models.model_evaluator import ModelEvaluator
from logger_config import get_logger

# =========================== FIXTURES =========================== #


@pytest.fixture
def sample_models_fixture(sample_data_fixture):
    """Returns a dictionary of trained real models for testing."""
    X_train, _, y_train, _ = sample_data_fixture  # Only use training data

    # Create real models
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    lr_model = LogisticRegression(max_iter=200, random_state=42)

    # Fit models to small sample data
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    return {"RandomForestClassifier": rf_model, "LogisticRegression": lr_model}


@pytest.fixture
def sample_data_fixture():
    """Returns sample feature and target datasets."""
    X_train = pd.DataFrame({"feature1": range(5), "feature2": range(5, 10)})
    X_test = pd.DataFrame({"feature1": range(5, 10), "feature2": range(10, 15)})
    y_train = pd.Series([0, 1, 0, 1, 0])
    y_test = pd.Series([1, 0, 1, 1, 0])

    return X_train, X_test, y_train, y_test


@pytest.fixture
def model_evaluator_fixture(sample_models_fixture, sample_data_fixture):
    """Returns a ModelEvaluator instance with mock data."""
    X_train, X_test, y_train, y_test = sample_data_fixture
    return ModelEvaluator(sample_models_fixture, X_train, X_test, y_train, y_test)


# =========================== TEST EVALUATION =========================== #


def test_evaluate_models(model_evaluator_fixture):
    """Test model evaluation and classification reports."""
    test_name = "test_evaluate_models"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        reports = model_evaluator_fixture.evaluate_models()

        assert "RandomForestClassifier" in reports
        assert "LogisticRegression" in reports

        assert "train_report" in reports["RandomForestClassifier"]
        assert "test_report" in reports["RandomForestClassifier"]

        assert isinstance(reports["RandomForestClassifier"]["train_report"], dict)
        assert isinstance(reports["RandomForestClassifier"]["test_report"], dict)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


# =========================== TEST ROC CURVE =========================== #


@patch("models.model_evaluator.RocCurveDisplay.from_estimator")
def test_plot_roc_curves(mock_roc_curve, model_evaluator_fixture):
    """Test that ROC curves are generated and stored."""
    test_name = "test_plot_roc_curves"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        model_evaluator_fixture.plot_roc_curves()
        assert "roc_curve" in model_evaluator_fixture.plots
        mock_roc_curve.assert_called()
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


# =========================== TEST SHAP EXPLANATION =========================== #


def test_plot_feature_importance(model_evaluator_fixture):
    """Test feature importance plotting."""
    test_name = "test_plot_feature_importance"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        model_evaluator_fixture.plot_feature_importance(
            "RandomForestClassifier", ["feature1", "feature2"]
        )
        assert (
            "feature_importance_RandomForestClassifier" in model_evaluator_fixture.plots
        )
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_plot_feature_importance_invalid_model(model_evaluator_fixture):
    """Test feature importance for a non-existent model raises ValueError."""
    test_name = "test_plot_feature_importance_invalid_model"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        with pytest.raises(ValueError, match="Model 'InvalidModel' not found"):
            model_evaluator_fixture.plot_feature_importance(
                "InvalidModel", ["feature1", "feature2"]
            )
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_plot_feature_importance_no_importance_attribute(model_evaluator_fixture):
    """Test when a model doesn't have feature importances."""
    test_name = "test_plot_feature_importance_no_importance_attribute"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        with pytest.raises(ValueError, match="does not support feature importance"):
            model_evaluator_fixture.plot_feature_importance(
                "LogisticRegression", ["feature1", "feature2"]
            )
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


# =========================== TEST PLOT SAVING =========================== #


def test_save_plots(model_evaluator_fixture, tmp_path):
    """Test saving plots to disk."""
    test_name = "test_save_plots"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        model_evaluator_fixture.plot_roc_curves()  # Generate a sample plot
        save_dir = tmp_path / "plots"
        model_evaluator_fixture.save_plots(str(save_dir))

        assert (save_dir / "roc_curve.png").exists()
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


# =========================== TEST MODEL SAVING =========================== #


def test_save_models(sample_models_fixture, tmp_path):
    """Test saving trained models."""
    test_name = "test_save_models"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        model_dir = tmp_path / "models"
        os.makedirs(model_dir)

        evaluator = ModelEvaluator(sample_models_fixture, None, None, None, None)
        evaluator.save_models(str(model_dir))

        assert os.path.exists(model_dir / "RandomForestClassifier.pkl")
        assert os.path.exists(model_dir / "LogisticRegression.pkl")
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_load_models(sample_models_fixture, tmp_path):
    """Test loading saved models."""
    test_name = "test_load_models"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        model_dir = tmp_path / "models"
        os.makedirs(model_dir)

        rf_model = RandomForestClassifier()
        lr_model = LogisticRegression()

        joblib.dump(rf_model, model_dir / "RandomForestClassifier.pkl")
        joblib.dump(lr_model, model_dir / "LogisticRegression.pkl")

        evaluator = ModelEvaluator(sample_models_fixture, None, None, None, None)
        evaluator.load_models(str(model_dir))

        assert isinstance(
            evaluator.models["RandomForestClassifier"], RandomForestClassifier
        )
        assert isinstance(evaluator.models["LogisticRegression"], LogisticRegression)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


# =========================== TEST JSON RESULTS SAVING =========================== #


def test_save_evaluation_results(model_evaluator_fixture, tmp_path):
    """Test saving evaluation results to a JSON file."""
    test_name = "test_save_evaluation_results"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        results = {"RandomForestClassifier": {"test_report": {}, "train_report": {}}}
        save_path = tmp_path / "evaluation.json"

        model_evaluator_fixture.save_evaluation_results(results, str(save_path))
        assert os.path.exists(save_path)

        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "RandomForestClassifier" in data
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise
