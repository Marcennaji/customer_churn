import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
import json
from sklearn.metrics import classification_report, RocCurveDisplay
from logger_config import logger


class ModelEvaluator:
    """Handles model evaluation, visualization, and feature importance reporting."""

    def __init__(
            self,
            models,
            X_train,
            X_test,
            y_train,
            y_test,
            model_names=None):
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
        self.model_names = model_names or {
            name: name for name in models.keys()}
        self.plots = {}

    def evaluate_models(self):
        """Evaluates models and returns classification reports."""
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

    def explain_shap(self, model_name):
        """Generates and stores a SHAP summary plot for a tree-based model."""
        if model_name not in self.models:
            raise ValueError(f"⚠ Model '{model_name}' not found in evaluator.")

        model = self.models[model_name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)

        shap.summary_plot(
            shap_values,
            self.X_test,
            plot_type="bar",
            show=False)
        fig = plt.gcf()
        self.plots[f"shap_{model_name}"] = fig

    def plot_feature_importance(self, model_name, feature_names):
        """Generates and stores a feature importance plot for a tree-based model."""
        if model_name not in self.models:
            raise ValueError(f"⚠ Model '{model_name}' not found in evaluator.")

        model = self.models[model_name]

        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                f"⚠ Model '{model_name}' does not support feature importance."
            )

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.set_title(
            f"Feature Importance - {self.model_names.get(model_name, model_name)}"
        )
        ax.set_ylabel("Importance")
        ax.bar(range(len(feature_names)), importances[indices])
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)

        self.plots[f"feature_importance_{model_name}"] = fig

    def save_plots(self, save_dir: str):
        """Saves all stored plots to the specified directory."""
        import os

        os.makedirs(save_dir, exist_ok=True)

        for name, fig in self.plots.items():
            file_path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(file_path, bbox_inches="tight", dpi=300)
            logger.info(f"Plot saved: {file_path}")

    def show_plots(self):
        """Displays all stored plots."""
        for fig in self.plots.values():
            fig.show()

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

    def save_evaluation_results(
        self, results, save_file_path="evaluation_results.json"
    ):
        """Saves evaluation results to a JSON file."""
        with open(save_file_path, "w") as f:
            json.dump(results, f, indent=4)
