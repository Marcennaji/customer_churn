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
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - [%(module)s] %(message)s",
            handlers=[
                logging.FileHandler(LOG_FILE_PATH, mode="a"),  # Append mode
                logging.StreamHandler(),
            ],
        )

        logger = logging.getLogger("ProjectLogger")
        logger.info("Logger successfully configured.")
        return logger

    except Exception as e:
        raise LoggerConfigurationError(f"Error configuring logger: {str(e)}") from e


# Initialize logger
logger = setup_logger()
