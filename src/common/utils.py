# filepath: /c:/Users/Utilisateur/dev/perso/devops/udacity/customer_churn/src/utils.py
import argparse
import os


def check_args_paths(description, config_help, csv_help, result_help):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=config_help,
    )
    parser.add_argument("--csv", type=str, required=True, help=csv_help)
    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help=result_help,
    )
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Check if CSV file exists
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    # Check if the directory of the result file exists
    result_dir = os.path.dirname(args.result)
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(
            f"Directory for result file does not exist: {result_dir}"
        )

    return args.config, args.csv, args.result
