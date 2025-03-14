"""
This module handles Exploratory Data Analysis (EDA) visualizations for the customer churn project.
Author: Marc Ennaji
Date: 2025-03-01
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger_config import get_logger


class EDAVisualizer:
    """A class for performing Exploratory Data Analysis (EDA) visualizations."""

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the EDAVisualizer with a dataset.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        get_logger().info("Initializing EDAVisualizer")
        self.df = df
        self.plots = {}  # Store figures for later use
        sns.set_style("whitegrid")

    def plot_histogram(self, column: str, figsize=(12, 6), bins=30, color="skyblue"):
        """Generates a histogram for a given column and stores it."""
        get_logger().info("Plotting histogram for column: %s", column)
        fig, ax = plt.subplots(figsize=figsize)

        unique_values = sorted(self.df[column].dropna().unique())
        is_binary = set(unique_values) <= {0, 1}

        if is_binary:
            bins = [0, 1, 2]
            ax.set_xticks([0, 1])
            counts = self.df[column].value_counts().sort_index()
            ax.bar(
                counts.index,
                counts.values,
                color=color,
                edgecolor="black",
                width=0.4,
                alpha=0.75,
            )
        else:
            self.df[column].hist(
                bins=bins, color=color, edgecolor="black", alpha=0.75, ax=ax
            )

        ax.set_title(f"Histogram of {column}", fontsize=14, fontweight="bold")
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        self.plots[f"histogram_{column}"] = fig

    def plot_bar_chart(
        self, column: str, normalize=True, figsize=(12, 6), palette="viridis"
    ):
        """Generates a bar chart for categorical values and stores it."""
        get_logger().info("Plotting bar chart for column: %s", column)
        fig, ax = plt.subplots(figsize=figsize)

        value_counts = self.df[column].value_counts(normalize=normalize)
        colors = sns.color_palette(palette, n_colors=len(value_counts))
        value_counts.plot(
            kind="bar", color=colors, edgecolor="black", alpha=0.85, ax=ax
        )
        ax.set_title(f"Bar Chart of {column}", fontsize=14, fontweight="bold")
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Proportion" if normalize else "Count", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        self.plots[f"bar_chart_{column}"] = fig

    def plot_kde(self, column: str, figsize=(12, 6), color="purple"):
        """Generates a Kernel Density Estimate (KDE) plot for a numerical column and stores it."""
        get_logger().info("Plotting KDE for column: %s", column)
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(
            self.df[column],
            stat="density",
            kde=True,
            color=color,
            edgecolor="black",
            alpha=0.75,
            ax=ax,
        )
        ax.set_title(f"KDE Plot of {column}", fontsize=14, fontweight="bold")
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        self.plots[f"kde_{column}"] = fig

    def plot_correlation_heatmap(self, figsize=(12, 8), cmap="coolwarm"):
        """Generates a heatmap of the correlation matrix and stores it."""
        get_logger().info("Plotting correlation heatmap")
        fig, ax = plt.subplots(figsize=figsize)
        numeric_df = self.df.select_dtypes(include=["number"])
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap=cmap,
            linewidths=1,
            fmt=".2f",
            linecolor="black",
            ax=ax,
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        self.plots["correlation_heatmap"] = fig

    def save_plots(self, output_dir: str):
        """Saves all stored plots to the specified directory."""
        get_logger().info("Saving plots to directory: %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for name, fig in self.plots.items():
            file_path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(file_path, bbox_inches="tight", dpi=300)
            get_logger().info("Plot saved: %s", file_path)

    def show_plots(self):
        """Displays all stored plots."""
        get_logger().info("Showing plots")
        for fig in self.plots.values():
            fig.show()
