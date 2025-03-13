import os
import fnmatch
import ast
import sys

# Set the console encoding to UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Define icons for different file types
ICONS = {"directory": "📁", "python": "🐍", "json": "📄", "image": "🖼️", "default": "📄"}


def get_module_docstring(filepath):
    """Extracts the module docstring from a Python file."""
    with open(filepath, "r", encoding="utf-8") as file:
        file_content = file.read()
        parsed_ast = ast.parse(file_content)
        docstring = ast.get_docstring(parsed_ast)
        if docstring:
            # Strip characters that might mess up with the Markdown formatting
            docstring = (
                docstring.replace("*", "")
                .replace("_", "")
                .replace("`", "")
                .replace("-", " ")
                .strip()
            )
        return docstring


def get_icon(entry):
    """Returns the appropriate icon for the given entry."""
    if os.path.isdir(entry):
        return ICONS["directory"]
    elif entry.endswith(".py"):
        return ICONS["python"]
    elif entry.endswith(".json"):
        return ICONS["json"]
    elif entry.endswith((".png", ".jpg", ".jpeg", ".gif")):
        return ICONS["image"]
    else:
        return ICONS["default"]


def generate_tree(startpath, indent="", descriptions={}, ignore_patterns=[]):
    description = descriptions.get(os.path.abspath(startpath), "")
    icon = get_icon(startpath)
    if description:
        print(f"{indent}{icon} **{os.path.basename(startpath)}** *{description}*")
    else:
        print(f"{indent}{icon} **{os.path.basename(startpath)}**")

    if os.path.isdir(startpath):
        for entry in sorted(os.listdir(startpath)):
            full_path = os.path.join(startpath, entry)
            if not any(fnmatch.fnmatch(entry, pattern) for pattern in ignore_patterns):
                if os.path.isdir(full_path) and not entry.startswith("."):
                    generate_tree(
                        full_path, indent + "    ", descriptions, ignore_patterns
                    )
                else:
                    if os.path.abspath(full_path) in descriptions:
                        desc = descriptions.get(os.path.abspath(full_path), "")
                        icon = get_icon(full_path)
                        if full_path.endswith(".py"):
                            docstring = get_module_docstring(full_path)
                            if docstring:
                                desc += f" - {docstring.splitlines()[0]}"
                        if desc:
                            print(f"{indent}    {icon} **{entry}** *{desc}*")
                        else:
                            print(f"{indent}    {icon} **{entry}**")


# Example usage with descriptions and ignore patterns
descriptions = {
    os.path.abspath("."): "Main project",
    os.path.abspath("./src"): "Source code directory",
    os.path.abspath("./src/eda"): "Exploratory Data Analysis module",
    os.path.abspath("./src/models"): "Model training and evaluation module",
    os.path.abspath(
        "./src/churn_library.py"
    ): "Module for loading and evaluating models",
    os.path.abspath("./utils"): "Utility functions directory",
    os.path.abspath("./data"): "Directory containing data files",
    os.path.abspath("./config"): "Directory containing configuration files",
    os.path.abspath("./config/config.json"): "Main configuration file",
    os.path.abspath("./config/preprocessing.json"): "Preprocessing configuration file",
    os.path.abspath("./config/splitting.json"): "Data splitting configuration file",
    os.path.abspath("./config/training.json"): "Model training configuration file",
    os.path.abspath("./results"): "Directory containing the results of a pipeline run",
    os.path.abspath("./models"): "Directory containing trained models",
    os.path.abspath("./tests"): "Directory containing unit tests scripts",
}

ignore_patterns = [
    "__pycache__",
    "*venv",
    ".vscode",
    "*.tmp",
    "*.bak",
    "__pycache*",
    "__init__.py",
    "customer_churn.egg-info",
    ".git*",
    "*_cache*",
    ".ipynb_checkpoints",
]  # List of directory patterns to ignore

generate_tree(".", descriptions=descriptions, ignore_patterns=ignore_patterns)
