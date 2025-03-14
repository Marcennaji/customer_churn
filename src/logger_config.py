"""
This module configures and provides a logger for the Customer Churn project.

The logger is configured with both file and stream handlers to log messages to a file and the console.
The default log file is located in the `../logs/customer_churn.log` relative to this module's directory.

Functions:
    setup_logger(log_file=None): Configures the logger with file and stream handlers.
    get_logger(): Returns the configured logger, initializing it if necessary.

Constants:
    DEFAULT_LOG_FILE (str): The default path to the log file.

Exceptions:
    LoggerConfigurationError: Raised when there is an error configuring the logger.
"""

import logging
import os
from common.exceptions import LoggerConfigurationError

DEFAULT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../logs/customer_churn.log"
)

_logger = None  # Holds the logger instance


def setup_logger(log_file=None):
    """Configures the project logger with file and stream handlers."""
    global _logger

    try:
        log_file = log_file or DEFAULT_LOG_FILE
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger("CustomerChurn")
        logger.setLevel(logging.INFO)

        # Remove existing handlers to prevent duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(module)s] %(message)s"
        )

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info("Logger initialized. Writing logs to: %s", log_file)

        _logger = logger  # Store the configured logger
        return _logger

    except Exception as e:
        raise LoggerConfigurationError(f"Error configuring logger: {str(e)}") from e


def get_logger():
    """Returns the configured logger, initializing it if necessary."""
    if _logger is None:
        return setup_logger()  # use default log file
    return _logger
