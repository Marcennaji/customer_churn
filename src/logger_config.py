"""
This module configures the project logger with file and stream handlers.
Author: Marc Ennaji
Date: 2025-03-13
"""

import logging
import os
from common.exceptions import LoggerConfigurationError

LOG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../logs/customer_churn.log"
)


def setup_logger():
    """Configures the project logger with file and stream handlers."""
    try:
        # Ensure the log directory exists
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(module)s] %(message)s",
            handlers=[
                logging.FileHandler(LOG_FILE_PATH, mode="w"),
                logging.StreamHandler(),
            ],
        )

        log = logging.getLogger("ProjectLogger")
        log.info("Logger successfully configured.")
        return log

    except Exception as e:
        raise LoggerConfigurationError(f"Error configuring logger: {str(e)}") from e


# Initialize logger
logger = setup_logger()
