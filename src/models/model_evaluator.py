"""
This module handles model evaluation, visualization, and feature importance reporting for the customer churn project.
Author: Marc Ennaji
Date: 2025-03-01
"""

import json
import os
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import classification_report, RocCurveDisplay
from logger_config import get_logger


class ModelEvaluator:
    """Handles model evaluation, visualization, and feature importance reporting."""

    def __init__(self, models, X_train, X_test, y_train, y_test, model_names=None):
        """
        Initializes the ModelEvaluator.

        Args:
            models (dict): Dictionary of trained models {"model_name": model_instance}.
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.
            model_names (dict, optional): Mapping of model keys to display names.
        """
        get_logger().info("Initializing ModelEvaluator")
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_names = model_names or {name: name for name in models.keys()}
        self.plots = {}

    def evaluate_models(self):
        """Evaluates models and returns classification reports."""
        get_logger().info("Evaluating models")
        reports = {}

        for name, model in self.models.items():
            y_train_preds = model.predict(self.X_train)
            y_test_preds = model.predict(self.X_test)

            reports[name] = {
                "train_report": classification_report(
                    self.y_train, y_train_preds, output_dict=True
                ),
                "test_report": classification_report(
                    self.y_test, y_test_preds, output_dict=True
                ),
            }

        return reports  # The caller can print, log, or save this data

    def plot_roc_curves(self):
        """Generates and stores ROC curves."""
        get_logger().info("Plotting ROC curves")
        fig, ax = plt.subplots(figsize=(15, 8))

        for name, model in self.models.items():
            RocCurveDisplay.from_estimator(
                model,
                self.X_test,
                self.y_test,
                ax=ax,
                alpha=0.8,
                name=self.model_names.get(name, name),
            )

        ax.set_title("ROC Curves")
        self.plots["roc_curve"] = fig

    def plot_feature_importance(self, model_name, feature_names):
        """Generates and stores a feature importance plot for a tree-based model."""
        get_logger().info("Plotting feature importance for model: %s", model_name)
        if model_name not in self.models:
            message = "Model '%s' not found in evaluator." % model_name
            get_logger().error(message)
            raise ValueError(message)

        model = self.models[model_name]

        if not hasattr(model, "feature_importances_"):
            message = "Model '%s' does not support feature importance." % model_name
            get_logger().error(message)
            raise ValueError(message)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.set_title(
            "Feature Importance - %s" % self.model_names.get(model_name, model_name)
        )
        ax.set_ylabel("Importance")
        ax.bar(range(len(feature_names)), importances[indices])
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)

        self.plots["feature_importance_%s" % model_name] = fig

    def save_plots(self, save_dir: str):
        """Saves all stored plots to the specified directory."""
        get_logger().info("Saving plots to directory: %s", save_dir)
        os.makedirs(save_dir, exist_ok=True)

        for name, fig in self.plots.items():
            file_path = os.path.join(save_dir, "%s.png" % name)
            fig.savefig(file_path, bbox_inches="tight", dpi=300)
            get_logger().info("Plot saved: %s", file_path)

    def show_plots(self):
        """Displays all stored plots."""
        get_logger().info("Showing plots")
        for fig in self.plots.values():
            fig.show()

    def save_models(self, save_dir="./models"):
        """Saves all models to disk."""
        get_logger().info("Saving models to directory: %s", save_dir)
        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.models.items():
            model_path = os.path.join(save_dir, "%s.pkl" % name)
            joblib.dump(model, model_path)
            get_logger().info("Model saved: %s", model_path)

    def load_models(self, load_dir="./models"):
        """Loads models from disk."""
        get_logger().info("Loading models from directory: %s", load_dir)
        for name in self.models.keys():
            model_path = os.path.join(load_dir, "%s.pkl" % name)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                get_logger().info("Model loaded: %s", model_path)

    def save_evaluation_results(
        self, results, save_file_path="evaluation_results.json"
    ):
        """Saves evaluation results to a JSON file."""
        get_logger().info("Saving evaluation results to file: %s", save_file_path)
        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            get_logger().info("Evaluation results saved: %s", save_file_path)

    def get_model_names(self):
        """Returns the mapping of model names."""
        get_logger().info("Getting model names")
        return self.model_names
