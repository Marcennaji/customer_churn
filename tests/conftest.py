"""
This module configures the logger for the test suite.
Author: Marc Ennaji
Date: 2025-03-01
"""

import os
import pytest
from logger_config import setup_logger


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Reconfigures the logger to use a separate log file for tests."""
    try:
        # Get the absolute path to the logs directory
        logs_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "logs")
        )
        os.makedirs(logs_dir, exist_ok=True)  # Ensure the logs directory exists

        # Define the absolute path to the log file
        log_file = os.path.join(logs_dir, "customer_churn_tests.log")
        print(f"Log file path: {log_file}")  # Debugging: Print the log file path

        # Configure the logger
        setup_logger(log_file=log_file)
    except Exception as e:
        print(f"Error setting up logger: {e}")
