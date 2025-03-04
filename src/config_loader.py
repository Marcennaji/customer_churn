import json
import os
import logging

def load_config(json_path):
    """Loads configuration from a JSON file."""
    if not os.path.exists(json_path):
        logging.error(f"Configuration file '{json_path}' not found.")
        return {}

    try:
        with open(json_path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file '{json_path}'. Check its format.")
        return {}
