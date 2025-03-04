# Customer Churn Prediction (Work in Progress)

## Description
Udacity MLOPS engineer Nanodegree.
This project implements an MLOps pipeline for customer churn prediction. It includes data collection, preprocessing, model training, and evaluation.

## Project Architecture
```
customer_churn
    │   pytest.ini
    │   README.md
    │   requirements.txt
    │   setup.py
    │
    ├───config
    │       bank_data_cleaner_config.json
    │       bank_data_encoder_config.json
    │       config.json
    │
    ├───data
    │       bank_data.csv
    │
    ├───images
    │   ├───eda
    │   └───results
    ├───logs
    │       customer_churn.log
    │
    ├───models
    │       logistic_model.pkl
    │       rfc_model.pkl
    │
    ├───src
    │   │   churn_library.py
    │   │   config_loader.py
    │   │   logger_config.py
    │   │
    │   ├───common
    │   │       exceptions.py
    │   │
    │   ├───data_processing
    │   │       data_cleaner.py
    │   │       data_encoder.py
    │   │       data_explorer.py
    │   │       encoder_base.py
    │   │       label_encoder.py
    │   │       one_hot_encoder.py
    │   │       ordinal_encoder.py
    │   │
    │   ├───models
    │   │       model_trainer.py
    │   │
    │   └───notebooks
    │           churn_notebook.ipynb
    │           Guide.ipynb
    │
    └───tests
            churn_script_logging_and_tests.py
            test_data_cleaner.py
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
```

## Usage
See churn_library.py for an example of a complete ML pipeline.
Execute the pipeline on a sample dataset, with:
```bash
churn_library
```
## Tests
Run unit tests:
```bash
pytest
```

## Author
- **Marc Ennaji** 

## License
This project is licensed under the MIT License.

