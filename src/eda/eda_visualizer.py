import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from common.utils import check_args_paths
from logger_config import logger
from data_preprocessing.data_cleaner import DataCleaner


class EDAVisualizer:
    """A class for performing Exploratory Data Analysis (EDA) visualizations."""

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the EDAVisualizer with a dataset.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        self.df = df

    def plot_histogram(self, column: str, figsize=(20, 10), bins=30):
        """Plots a histogram for a given column."""
        plt.figure(figsize=figsize)
        self.df[column].hist(bins=bins)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_bar_chart(self, column: str, normalize=True, figsize=(20, 10)):
        """Plots a bar chart for categorical values."""
        plt.figure(figsize=figsize)
        self.df[column].value_counts(normalize=normalize).plot(kind="bar")
        plt.title(f"Bar Chart of {column}")
        plt.xlabel(column)
        plt.ylabel("Proportion" if normalize else "Count")
        plt.show()

    def plot_kde(self, column: str, figsize=(20, 10)):
        """Plots a Kernel Density Estimate (KDE) plot for a numerical column."""
        plt.figure(figsize=figsize)
        sns.histplot(self.df[column], stat="density", kde=True)
        plt.title(f"KDE Plot of {column}")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.show()

    def plot_correlation_heatmap(self, figsize=(20, 10), cmap="Dark2_r"):
        """Plots a heatmap of the correlation matrix."""
        plt.figure(figsize=figsize)
        sns.heatmap(self.df.corr(), annot=False, cmap=cmap, linewidths=2)
        plt.title("Correlation Heatmap")
        plt.show()


def main():
    try:
        config_path, csv_path, result_path = check_args_paths(
            description="Clean a dataset, then perform an EDA on the resulting dataframe.",
            config_help="Path to the JSON configuration file.",
            csv_help="Path to the input dataset CSV file.",
            result_help="Directory path to save the generated images",
        )
    except FileNotFoundError as e:
        logger.error(e)
        print(e)
        return

    df = pd.read_csv(csv_path)

    cleaner = DataCleaner(config_json_file=config_path)
    cleaned_df = cleaner.clean_data(df)

    eda = EDAVisualizer(cleaned_df)

    eda.plot_histogram("churn")
    eda.plot_histogram("age")

    eda.plot_bar_chart("marital_status")
    eda.plot_kde("total_transaction_count")

    # eda.plot_correlation_heatmap()


if __name__ == "__main__":
    main()
