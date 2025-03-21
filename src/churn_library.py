"""
This module serves as the main pipeline for the customer churn project, handling data processing, model training, and evaluation.
Author: Marc Ennaji
Date: 2025-03-01
"""

import os
import argparse
import pandas as pd
import joblib

from data_preprocessing.data_cleaner import DataCleaner
from data_preprocessing.data_encoder import DataEncoder
from eda.eda_visualizer import EDAVisualizer
from models.data_splitter import DatasetSplitter
from models.model_trainer import ModelTrainer
from models.model_evaluator import ModelEvaluator
from config_manager import ConfigManager
from common.exceptions import MLPipelineError, ModelLoadError
from logger_config import get_logger


# ========================= CONFIGURATION LOADING ========================= #


def load_configs(config_file_path):
    """Loads all configuration files."""
    config_manager = ConfigManager(config_file_path)
    return {
        "csv_path": config_manager.get_csv_path(),
        "data_dir": config_manager.get_data_dir(),
        "models_dir": config_manager.get_models_dir(),
        "results_dir": config_manager.get_results_dir(),
        "preprocessing": config_manager.get_config("preprocessing"),
        "splitting": config_manager.get_config("splitting"),
        "training": config_manager.get_config("training"),
        "eval_only": config_manager.is_eval_only(),
    }


# ========================= DATA PROCESSING ========================= #


def import_and_clean_data(csv_path, preprocessing_config):
    """Imports, cleans, and preprocesses raw data."""
    df = pd.read_csv(csv_path)
    cleaner = DataCleaner(config=preprocessing_config)
    return cleaner.clean_data(
        df, drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
    )


def perform_eda(df):
    """Performs exploratory data analysis on cleaned data."""
    eda = EDAVisualizer(df)

    plots = {
        "bank_histo_churn.png": ("plot_histogram", {"column": "churn"}),
        "bank_histo_age.png": ("plot_histogram", {"column": "age"}),
        "bank_bar_marital_status.png": ("plot_bar_chart", {"column": "marital_status"}),
        "bank_kde_total_transaction_count.png": (
            "plot_kde",
            {"column": "total_transaction_count"},
        ),
        "bank_correlation_heatmap.png": ("plot_correlation_heatmap", {}),
    }

    for _, (method, kwargs) in plots.items():
        getattr(eda, method)(**kwargs)

    eda.save_plots("./results/images/eda")

    return eda


def encode_features(df, preprocessing_config):
    """Encodes categorical features."""
    encoder = DataEncoder(config=preprocessing_config)
    return encoder.encode(df)


def split_data(df, splitting_config):
    """Splits the dataset into training and test sets."""
    return DatasetSplitter(df, config=splitting_config, profile="default").split()


# ========================= MODEL TRAINING & EVALUATION ========================= #


def train_models(X_train, y_train, training_config, models_dir):
    """Trains models using the provided configuration."""
    trainer = ModelTrainer(training_config=training_config)
    models = trainer.train(X_train, y_train)
    trainer.save_models(models, models_dir)
    return models


def evaluate_models(models, config, X_train, X_test, y_train, y_test):
    """Evaluates trained models and generates reports and visualizations."""
    evaluator = ModelEvaluator(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        model_names={
            "RandomForestClassifier": "Random Forest",
            "LogisticRegression": "Logistic Regression",
        },
    )

    reports = evaluator.evaluate_models()
    evaluator.save_evaluation_results(
        reports,
        save_file_path=os.path.join(config["results_dir"], "json/evaluation.json"),
    )

    evaluator.plot_roc_curves()
    evaluator.plot_feature_importance(
        "RandomForestClassifier", X_train.columns.tolist()
    )
    evaluator.save_plots(save_dir=os.path.join(config["results_dir"], "images"))

    return evaluator


def load_models(models_to_load, models_dir):
    """Loads specified models from disk."""
    loaded_models = {}
    for model_name in models_to_load:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            loaded_models[model_name] = joblib.load(model_path)
            get_logger().info("Loaded model: %s", model_name)
        else:
            raise ModelLoadError(f"Model file not found: {model_path}")
    return loaded_models


# ========================= MAIN PIPELINE ========================= #


def main(config_file_path_arg):
    """
    Main pipeline execution function.
    """
    try:
        # Load configurations
        config = load_configs(config_file_path_arg)

        # Data processing
        df_cleaned = import_and_clean_data(config["csv_path"], config["preprocessing"])
        perform_eda(df_cleaned)
        df_encoded = encode_features(df_cleaned, config["preprocessing"])

        # Data splitting
        X_train, X_test, y_train, y_test = split_data(df_encoded, config["splitting"])

        # Load models if evaluation-only mode is enabled
        if config["eval_only"]:
            models = load_models(
                ["RandomForestClassifier", "LogisticRegression"], config["models_dir"]
            )
        else:
            models = train_models(
                X_train, y_train, config["training"], config["models_dir"]
            )

        # Model evaluation
        evaluate_models(models, config, X_train, X_test, y_train, y_test)

        get_logger().info("Pipeline execution completed successfully")

    except MLPipelineError as e:
        get_logger().error("%s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the churn prediction pipeline.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config.json file"
    )
    args = parser.parse_args()
    caller_dir = os.getcwd()
    config_file = os.path.abspath(os.path.join(caller_dir, args.config))
    main(config_file)
