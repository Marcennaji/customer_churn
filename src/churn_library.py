from data_preprocessing.data_cleaner import DataCleaner
from eda.eda_visualizer import EDAVisualizer
from models.data_splitter import DatasetSplitter
from data_preprocessing.data_encoder import DataEncoder
from models.model_trainer import ModelTrainer
import pandas as pd
from config_manager import ConfigManager
import os
from logger_config import logger
from common.exceptions import MLPipelineError
from models.model_evaluator import ModelEvaluator

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def import_data(pth):
    """Loads CSV into a DataFrame without modifications."""
    return pd.read_csv(pth)


def perform_eda(df):
    """Performs exploratory data analysis on cleaned data."""
    eda = EDAVisualizer(df)

    eda.plot_histogram("churn", os.path.join(IMAGES_DIR, "bank_histo_churn.png"))
    eda.plot_histogram("age", os.path.join(IMAGES_DIR, "bank_histo_age.png"))
    eda.plot_bar_chart(
        "marital_status", os.path.join(IMAGES_DIR, "bank_bar_marital_status.png")
    )
    eda.plot_kde(
        "total_transaction_count",
        os.path.join(IMAGES_DIR, "bank_kde_total_transaction_count.png"),
    )
    eda.plot_correlation_heatmap(
        os.path.join(IMAGES_DIR, "bank_correlation_heatmap.png")
    )


def encoder_helper(df, config):
    """Encodes categorical features based on target variable proportions."""
    # encode categorical features
    encoder = DataEncoder(config=config)
    df_encoded = encoder.encode(df)
    return df_encoded


def main():

    try:
        config_manager = ConfigManager(description="ML Pipeline Configuration")

        csv_path = config_manager.get_csv_path()
        data_dir = config_manager.get_data_dir()
        models_dir = config_manager.get_models_dir()

        # Retrieve configurations
        preprocessing_config = config_manager.get_config("preprocessing")
        splitting_config = config_manager.get_config("splitting")
        training_config = config_manager.get_config("training")

        # Load raw data
        df_raw = import_data(csv_path)

        # Clean data : rename columns, drop columns, fill missing values, remove empty rows, replace categorical values
        cleaner = DataCleaner(config=preprocessing_config)
        df_cleaned = cleaner.clean_data(
            df_raw, drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
        )

        # perform_eda(df_cleaned)

        df_encoded = encoder_helper(df_cleaned, config=preprocessing_config)

        df_encoded.to_csv(
            os.path.join(data_dir, "processed/encoded_bank_data.csv"), index=False
        )

        splitter = DatasetSplitter(
            df_encoded, config=splitting_config, profile="default"
        )
        X_train, X_test, y_train, y_test = splitter.split()

        trainer = ModelTrainer(training_config=training_config)
        trained_models = trainer.train(X_train, y_train)

        for _, model in trained_models.items():
            ModelEvaluator.evaluate(model, X_train, X_test, y_train, y_test)

        trainer.save_models(trained_models, models_dir)

    except MLPipelineError as e:
        logger.error(e)
        print(e)


if __name__ == "__main__":
    main()
