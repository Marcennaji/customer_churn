import os
import fnmatch
import ast


def get_module_docstring(filepath):
    """Extracts the module docstring from a Python file."""
    with open(filepath, "r", encoding="utf-8") as file:
        file_content = file.read()
        parsed_ast = ast.parse(file_content)
        docstring = ast.get_docstring(parsed_ast)
        return docstring


def generate_tree(startpath, indent="", descriptions={}, ignore_patterns=[]):
    description = descriptions.get(startpath, "")
    print(f"{indent}- {os.path.basename(startpath)} {description}")

    if os.path.isdir(startpath):
        for entry in sorted(os.listdir(startpath)):
            full_path = os.path.join(startpath, entry)
            if not any(fnmatch.fnmatch(entry, pattern) for pattern in ignore_patterns):
                if os.path.isdir(full_path) and not entry.startswith("."):
                    generate_tree(
                        full_path, indent + "  ", descriptions, ignore_patterns
                    )
                else:
                    desc = descriptions.get(full_path, "")
                    if full_path.endswith(".py"):
                        docstring = get_module_docstring(full_path)
                        if docstring:
                            desc += f" - {docstring.splitlines()[0]}"
                    print(f"{indent}  - {entry} {desc}")


# Example usage with descriptions and ignore patterns
descriptions = {
    ".": " - Projet principal",
    "./src": " - Code source",
    "./docs": " - Documentation",
    "./README.md": " - Fichier README principal",
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
