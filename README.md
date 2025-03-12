# Customer Churn Prediction (Work in Progress)

## Description
Udacity MLOps Engineer Nanodegree – February 2025

This project aim to refactor the src/notebooks/churn_notebook.ipynb file, by applying clean code best practices. See src/notebooks/Guide.jpynb for details.

The proposed solution aim to be flexible, generic, maintenable and easily configurable for different purposes, by providing a set of ML classes that goes beyond the specific problematic of a churn detection.

## Project Architecture
```
- .  - Projet principal
  - .pre-commit-config.yaml 
  - .workspace-config.json 
  - README.md 
  - config 
    - data_splitting_profiles.json 
    - preprocessing_config.json 
    - training_config.json 
  - config.json 
  - data 
    - processed 
    - raw 
      - bank_data.csv 
  - images 
  - logs 
    - customer_churn.log 
  - models 
    - LogisticRegression.pkl 
    - RandomForestClassifier.pkl 
  - pytest.ini 
  - requirements.txt 
  - results 
    - images 
      - eda 
        - bar_chart_marital_status.png 
        - correlation_heatmap.png 
        - histogram_age.png 
        - histogram_churn.png 
        - kde_total_transaction_count.png 
      - feature_importance_RandomForestClassifier.png 
      - roc_curve.png 
      - shap_RandomForestClassifier.png 
    - json 
      - evaluation.json 
  - src 
    - churn_library.py  - This module serves as the main pipeline for the customer churn project, handling data processing, model training, and evaluation.
    - common 
      - exceptions.py  - This module defines custom exceptions for the customer churn project.
    - config_manager.py 
    - data_preprocessing 
      - data_cleaner.py  - This module handles general data cleaning operations for the customer churn project.
      - data_encoder.py  - This module handles categorical feature encoding based on JSON configuration for the customer churn project.
      - encoder_type.py  - This module provides an abstract base class for categorical encoders in the customer churn project.
    - eda 
      - eda_visualizer.py  - This module handles Exploratory Data Analysis (EDA) visualizations for the customer churn project.
    - logger_config.py 
    - models 
      - data_splitter.py  - This module handles train-test data splitting based on JSON configuration profiles for the customer churn project.
      - model_evaluator.py  - This module handles model evaluation, visualization, and feature importance reporting for the customer churn project.
      - model_trainer.py  - This module handles the training and hyperparameter tuning of models for the customer churn project.
    - notebooks 
      - Guide.ipynb 
      - churn_notebook.ipynb 
      - churn_notebook.py 
    - results 
      - images 
        - eda 
          - bar_chart_marital_status.png 
          - correlation_heatmap.png 
          - histogram_age.png 
          - histogram_churn.png 
          - kde_total_transaction_count.png 
  - test.md 
  - tests 
    - test_data_cleaner.py 
    - test_data_encoder.py 
    - test_data_splitter.py 
    - test_eda_visualizer.py 
    - test_model_evaluator.py 
    - test_model_trainer.py 
  - utils 
    - generate_doc_tree.py 
    - pylint_checker.out 
    - pylint_checker.py 
    - test.md 

```

## Prerequisites
- Python 3.10+
- Libraries: see `requirements.txt`

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Marcennaji/customer_churn.git
cd customer_churn
pip install -r requirements.txt
pip install -e .
```

## Usage
See the python script `churn_library.py`, for an example of a complete ML pipeline.
For executing the ML pipeline on a sample dataset with default parameters values, run:
```bash
churn_library --csv=data/raw/bank_data.csv
```
You can also override parameters default values, for example:
```bash
churn_library --preprocessing-config=config/preprocessing_config.json --splitting-config=config/data_splitting_profiles.json --training-config=config/training_config.json  --csv=data/raw/bank_data.csv --data-dir=data --models-dir=models
```
## Tests
Run unit tests:
```bash
pytest
```

## Author
**Marc Ennaji** 

## License
This project is licensed under the MIT License.

