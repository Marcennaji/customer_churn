"""
This module contains unit tests for the EDAVisualizer class, which handles EDA visualizations.
Author: Marc Ennaji
Date: 2025-03-01

Disabled the W0621 pylint warning, as it triggers a false positive when using fixtures (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint) (see https://stackoverflow.com/questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint)

"""

# pylint: disable=W0621

from unittest.mock import patch
import pytest
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from eda.eda_visualizer import EDAVisualizer
from logger_config import get_logger

matplotlib.use("Agg")


@pytest.fixture
def sample_df_fixture():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "numeric_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "binary_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "categorical_col": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def eda_visualizer_fixture(sample_df_fixture):
    """Fixture to provide an instance of EDAVisualizer for testing."""
    return EDAVisualizer(sample_df_fixture)


def test_initialization(sample_df_fixture):
    """Test the initialization of EDAVisualizer."""
    test_name = "test_initialization"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        visualizer = EDAVisualizer(sample_df_fixture)
        assert visualizer.df.equals(sample_df_fixture)
        assert not visualizer.plots
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_plot_histogram(eda_visualizer_fixture):
    """Test the plot_histogram method."""
    test_name = "test_plot_histogram"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        eda_visualizer_fixture.plot_histogram("numeric_col")
        assert "histogram_numeric_col" in eda_visualizer_fixture.plots
        assert isinstance(
            eda_visualizer_fixture.plots["histogram_numeric_col"], plt.Figure
        )
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_plot_bar_chart(eda_visualizer_fixture):
    """Test the plot_bar_chart method."""
    test_name = "test_plot_bar_chart"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        eda_visualizer_fixture.plot_bar_chart("categorical_col")
        assert "bar_chart_categorical_col" in eda_visualizer_fixture.plots
        assert isinstance(
            eda_visualizer_fixture.plots["bar_chart_categorical_col"], plt.Figure
        )
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_plot_kde(eda_visualizer_fixture):
    """Test the plot_kde method."""
    test_name = "test_plot_kde"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        eda_visualizer_fixture.plot_kde("numeric_col")
        assert "kde_numeric_col" in eda_visualizer_fixture.plots
        assert isinstance(eda_visualizer_fixture.plots["kde_numeric_col"], plt.Figure)
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


def test_plot_correlation_heatmap(eda_visualizer_fixture):
    """Test the plot_correlation_heatmap method."""
    test_name = "test_plot_correlation_heatmap"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        eda_visualizer_fixture.plot_correlation_heatmap()
        assert "correlation_heatmap" in eda_visualizer_fixture.plots
        assert isinstance(
            eda_visualizer_fixture.plots["correlation_heatmap"], plt.Figure
        )
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise


@patch("os.makedirs")
@patch("matplotlib.figure.Figure.savefig")
def test_save_plots(mock_savefig, mock_makedirs, eda_visualizer_fixture):
    """Test the save_plots method."""
    test_name = "test_save_plots"
    get_logger().info("**********  RUNNING %s  **********", test_name)
    try:
        eda_visualizer_fixture.plot_histogram("numeric_col")
        eda_visualizer_fixture.plot_bar_chart("categorical_col")
        eda_visualizer_fixture.save_plots("output_dir")
        assert mock_savefig.call_count == 2
    except AssertionError as e:
        get_logger().error("Assertion failed in %s: %s", test_name, str(e))
        raise
