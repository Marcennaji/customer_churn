from setuptools import setup, find_packages

setup(
    name="customer_churn",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "churn_library=churn_library:main",
        ],
    },
)
