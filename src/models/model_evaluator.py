import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
import json
from sklearn.metrics import classification_report, RocCurveDisplay


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
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_names = model_names or {name: name for name in models.keys()}

    def evaluate_models(self):
        """
        Evaluates models and returns classification reports.

        Returns:
            dict: {model_name: {"train_report": {...}, "test_report": {...}}}
        """
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

    def plot_roc_curves(self, save_path=None):
        """
        Generates ROC curves and optionally saves the plot.

        Args:
            save_path (str, optional): Path to save the plot. If None, only generates the plot.
        """
        plt.figure(figsize=(15, 8))
        ax = plt.gca()

        for name, model in self.models.items():
            RocCurveDisplay.from_estimator(
                model,
                self.X_test,
                self.y_test,
                ax=ax,
                alpha=0.8,
                name=self.model_names.get(name, name),
            )

        plt.title("ROC Curves")
        if save_path:
            plt.savefig(save_path)

    def explain_shap(self, model_name, save_path=None):
        """
        Generates a SHAP summary plot for a tree-based model.

        Args:
            model_name (str): Name of the model to explain.
            save_path (str, optional): Path to save the SHAP plot.
        """
        if model_name not in self.models:
            raise ValueError(f"⚠ Model '{model_name}' not found in evaluator.")

        model = self.models[model_name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)

        shap.summary_plot(
            shap_values, self.X_test, plot_type="bar", show=False
        )  # ⬅ Removed plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_feature_importance(self, model_name, feature_names, save_path=None):
        """
        Generates a feature importance plot for a tree-based model.

        Args:
            model_name (str): Name of the model to plot.
            feature_names (list): List of feature names.
            save_path (str, optional): Path to save the plot.
        """
        if model_name not in self.models:
            raise ValueError(f"⚠ Model '{model_name}' not found in evaluator.")

        model = self.models[model_name]

        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                f"⚠ Model '{model_name}' does not support feature importance."
            )

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(20, 5))
        plt.title(
            f"Feature Importance - {self.model_names.get(model_name, model_name)}"
        )
        plt.ylabel("Importance")
        plt.bar(range(len(feature_names)), importances[indices])
        plt.xticks(
            range(len(feature_names)), [feature_names[i] for i in indices], rotation=90
        )

        if save_path:
            plt.savefig(save_path)

    def save_models(self, save_dir="./models"):
        """Saves all models to disk."""
        import os

        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.models.items():
            model_path = f"{save_dir}/{name}.pkl"
            joblib.dump(model, model_path)

    def load_models(self, load_dir="./models"):
        """Loads models from disk."""
        import os

        for name in self.models.keys():
            model_path = f"{load_dir}/{name}.pkl"
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)

    def save_evaluation_results(self, results, save_path="evaluation_results.json"):
        """
        Saves evaluation results to a JSON file.

        Args:
            results (dict): Evaluation results from `evaluate_models()`.
            save_path (str): Path to save the JSON file.
        """
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
