import seaborn as sns
import matplotlib.pyplot as plt
import os
from logger_config import logger


class DataExplorer:
    """Handles exploratory data analysis (EDA) and visualization."""

    def __init__(self, output_dir="images/eda"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def perform_eda(self, df):
        """Generates EDA visualizations and saves them to disk."""
        logger.info("Performing Exploratory Data Analysis (EDA)...")

        # Histogram of numerical features
        for col in df.select_dtypes(include=["number"]).columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(self.output_dir, f"{col}_histogram.png"))
            plt.close()

        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
        plt.close()

        logger.info("EDA complete. Figures saved in images/eda/")
