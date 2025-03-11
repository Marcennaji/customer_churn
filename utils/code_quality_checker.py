import os
import subprocess
import argparse
import re


def format_code(file_path):
    """Formats the given Python file using autopep8."""
    try:
        subprocess.run(
            ["autopep8", "--in-place", "--aggressive", "--aggressive", file_path],
            check=True,
            timeout=60,
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
            timeout=60,
        )
        output = result.stdout.strip()

        # Extract the pylint score
        score = "N/A"
        for line in output.split("\n"):
            if "Your code has been rated at" in line:
                score = line.split("at")[-1].strip()
                break

        # Extract relevant pylint messages with line numbers
        output_lines = output.split("\n")
        filtered_output = []
        for line in output_lines:
            # Match with or without the file path prefix
            match = re.match(r"^(?:.*?:)?(\d+):\d+: (.+)", line)
            if match:
                line_number, message = match.groups()
                filtered_output.append(f"Ligne {line_number}: {message}")

        formatted_output = (
            "\n".join(filtered_output) if filtered_output else "Aucune erreur détectée."
        )

        return score, formatted_output

    except subprocess.TimeoutExpired:
        print(f"Analyzing {file_path} timed out.")
        return "N/A", "Timeout"
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing {file_path}: {e}")
        return "N/A", str(e)

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directories = [
        os.path.join(script_dir, "../src"),
        os.path.join(script_dir, "../tests"),
    ]

    report_data = []
    detailed_reports = []

    for root_directory in root_directories:
        print(f"Entering directory {root_directory}...")
        python_files = find_python_files(root_directory)

        for file in python_files:
            if "__init__.py" in file:
                continue
            print(f"Processing {file}...")

            format_code(file)
            score, detailed_output = analyze_code(file)

            relative_file = os.path.relpath(file, script_dir)

            # Store summary report
            report_data.append([relative_file, score])

            # Store detailed report
            detailed_reports.append((relative_file, score, detailed_output))

    # Write results to a file
    with open(args.report_file, "w", encoding="utf-8") as f:
        # Write tabular summary report
        f.write("SCORES SUMMARY\n\n")
        for row in report_data:
            f.write(f"{row[0]}\t{row[1]}\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write("DETAILED PYLINT REPORT\n\n")
        # Write detailed pylint reports
        for file, score, comments in detailed_reports:
            f.write(f"{file} (score: {score})\n")
            f.write(f"{'-' * len(file)}\n")
            f.write(comments + "\n\n")
            f.write("-" * 80 + "\n\n")

    print(f"Code quality report saved to {args.report_file}")


if __name__ == "__main__":
    main()
