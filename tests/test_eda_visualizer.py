import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from src.eda.eda_visualizer import EDAVisualizer


@pytest.fixture
def sample_df():
    data = {
        "numeric_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "binary_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "categorical_col": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def eda_visualizer(sample_df):
    return EDAVisualizer(sample_df)


def test_initialization(sample_df):
    visualizer = EDAVisualizer(sample_df)
    assert visualizer.df.equals(sample_df)
    assert visualizer.plots == {}


def test_plot_histogram(eda_visualizer):
    eda_visualizer.plot_histogram("numeric_col")
    assert "histogram_numeric_col" in eda_visualizer.plots
    assert isinstance(
        eda_visualizer.plots["histogram_numeric_col"],
        plt.Figure)


def test_plot_bar_chart(eda_visualizer):
    eda_visualizer.plot_bar_chart("categorical_col")
    assert "bar_chart_categorical_col" in eda_visualizer.plots
    assert isinstance(
        eda_visualizer.plots["bar_chart_categorical_col"],
        plt.Figure)


def test_plot_kde(eda_visualizer):
    eda_visualizer.plot_kde("numeric_col")
    assert "kde_numeric_col" in eda_visualizer.plots
    assert isinstance(eda_visualizer.plots["kde_numeric_col"], plt.Figure)


def test_plot_correlation_heatmap(eda_visualizer):
    eda_visualizer.plot_correlation_heatmap()
    assert "correlation_heatmap" in eda_visualizer.plots
    assert isinstance(eda_visualizer.plots["correlation_heatmap"], plt.Figure)


@patch("os.makedirs")
@patch("matplotlib.figure.Figure.savefig")
def test_save_plots(mock_savefig, mock_makedirs, eda_visualizer):
    eda_visualizer.plot_histogram("numeric_col")
    eda_visualizer.plot_bar_chart("categorical_col")
    eda_visualizer.save_plots("output_dir")
    assert mock_makedirs.called_once_with("output_dir", exist_ok=True)
    assert mock_savefig.call_count == 2
