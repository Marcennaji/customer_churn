import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config_manager import ConfigManager
from logger_config import logger
from data_preprocessing.data_cleaner import DataCleaner
import os
from common.exceptions import MLPipelineError


class EDAVisualizer:
    """A class for performing Exploratory Data Analysis (EDA) visualizations and saving plots to files."""

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the EDAVisualizer with a dataset.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        self.df = df
        sns.set_style("whitegrid")

    def _save_plot(self, file_path: str):
        """Saves the current plot to a file and logs the action."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Plot saved: {file_path}")

    def plot_histogram(
        self, column: str, file_path: str, figsize=(12, 6), bins=30, color="skyblue"
    ):
        """Plots and saves a histogram for a given column, adjusting for binary values."""
        plt.figure(figsize=figsize)

        # Detect if the column is binary (only contains 0 and 1)
        unique_values = sorted(self.df[column].dropna().unique())
        is_binary = set(unique_values) <= {0, 1}

        if is_binary:
            bins = [0, 1, 2]  # Force bins to only 0 and 1
            plt.xticks([0, 1])  # Ensure x-axis only shows 0 and 1

            # Use bar plot for better aesthetics (controlled width)
            counts = self.df[column].value_counts().sort_index()
            plt.bar(
                counts.index,
                counts.values,
                color=color,
                edgecolor="black",
                width=0.4,
                alpha=0.75,
            )
        else:
            # Regular histogram for non-binary data
            self.df[column].hist(bins=bins, color=color, edgecolor="black", alpha=0.75)

        plt.title(f"Histogram of {column}", fontsize=14, fontweight="bold")
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        self._save_plot(file_path)

    def plot_bar_chart(
        self,
        column: str,
        file_path: str,
        normalize=True,
        figsize=(12, 6),
        palette="viridis",
    ):
        """Plots and saves a bar chart for categorical values"""
        plt.figure(figsize=figsize)

        value_counts = self.df[column].value_counts(normalize=normalize)
        colors = sns.color_palette(palette, n_colors=len(value_counts))
        value_counts.plot(kind="bar", color=colors, edgecolor="black", alpha=0.85)
        plt.title(f"Bar Chart of {column}", fontsize=14, fontweight="bold")
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Proportion" if normalize else "Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        self._save_plot(file_path)

    def plot_kde(self, column: str, file_path: str, figsize=(12, 6), color="purple"):
        """Plots and saves a Kernel Density Estimate (KDE) plot for a numerical column."""
        plt.figure(figsize=figsize)
        sns.histplot(
            self.df[column],
            stat="density",
            kde=True,
            color=color,
            edgecolor="black",
            alpha=0.75,
        )
        plt.title(f"KDE Plot of {column}", fontsize=14, fontweight="bold")
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        self._save_plot(file_path)

    def plot_correlation_heatmap(
        self, file_path: str, figsize=(12, 8), cmap="coolwarm"
    ):
        """Plots and saves a heatmap of the correlation matrix, ensuring only numerical columns are considered."""
        plt.figure(figsize=figsize)
        numeric_df = self.df.select_dtypes(
            include=["number"]
        )  # Select only numerical columns
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap=cmap,
            linewidths=1,
            fmt=".2f",
            linecolor="black",
        )
        plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        self._save_plot(file_path)


def main():

    try:
        config_manager = ConfigManager(description="ML Pipeline Configuration")

        # Retrieve CSV and Result paths
        csv_path = config_manager.get_csv_path()

        # Retrieve configurations
        preprocessing_config = config_manager.get_config("preprocessing")

        # Load raw data
        df_raw = pd.read_csv(csv_path)

        # Clean data : rename columns, drop columns, fill missing values, remove empty rows, replace categorical values
        cleaner = DataCleaner(config=preprocessing_config)
        df_cleaned = cleaner.clean_data(
            df_raw, drop_columns=["CLIENTNUM"], fill_strategy="mean", remove_empty=True
        )

        eda = EDAVisualizer(df_cleaned)

        eda.plot_histogram("churn")
        eda.plot_histogram("age")
        eda.plot_bar_chart("marital_status")
        eda.plot_kde("total_transaction_count")
        eda.plot_correlation_heatmap()

    except MLPipelineError as e:
        logger.error(e)
        print(e)


if __name__ == "__main__":
    main()
