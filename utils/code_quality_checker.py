import os
import subprocess
import argparse
import csv


def format_code(file_path):
    """Formats the given Python file using autopep8."""
    try:
        subprocess.run(
            ["autopep8", "--in-place", "--aggressive", "--aggressive", file_path],
            check=True,
            timeout=60,  # Add a timeout of 60 seconds
        )
    except subprocess.TimeoutExpired:
        print(f"Formatting {file_path} timed out.")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting {file_path}: {e}")


def analyze_code(file_path):
    """Runs pylint on the given Python file and returns its score and output."""
    try:
        result = subprocess.run(
            ["pylint", file_path],
            capture_output=True,
            text=True,
            timeout=60,  # Add a timeout of 60 seconds
        )
        output = result.stdout.strip()

        # Extract the pylint score from the last line of the output
        score = "N/A"
        for line in output.split("\n"):
            if line.startswith("Your code has been rated at"):
                score = line.split("at")[-1].strip()
                break

        return score, output.replace("\n", "\t")  # Format for tabular output
    except subprocess.TimeoutExpired:
        print(f"Analyzing {file_path} timed out.")
        return "N/A", "Timeout"
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing {file_path}: {e}")
        return "N/A", str(e)


def find_python_files(directory):
    """Recursively finds all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def main():
    parser = argparse.ArgumentParser(
        description="Check Python code quality with autopep8 and pylint."
    )
    parser.add_argument(
        "report_file",
        nargs="?",
        default="code_quality_checker.out",
        help="Output report file (default: code_quality_checker.out)",
    )
    args = parser.parse_args()

    # Define the list of root directories relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directories = [
        os.path.join(script_dir, "../src"),
        os.path.join(script_dir, "../tests"),
    ]

    report_data = []
    for root_directory in root_directories:
        print(f"Entering in directory {root_directory}...")
        python_files = find_python_files(root_directory)
        for file in python_files:
            print(f"Processing {file}...")
            format_code(file)
            score, _ = analyze_code(file)
            report_data.append([file, score])

    # Write results to a file
    with open(args.report_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["script_name", "pylint_score"])
        writer.writerows(report_data)

    print(f"Code quality report saved to {args.report_file}")


if __name__ == "__main__":
    main()
