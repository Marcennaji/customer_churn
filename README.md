# Customer Churn Prediction (Work in Progress)

## Description
Udacity MLOPS engineer Nanodegree, feb. 2025.
This project implements an MLOps pipeline for customer churn prediction. It includes data collection, preprocessing, model training, and evaluation.

## Project Architecture
```
root
│   pytest.ini
│   README.md
│   requirements.txt
│   setup.py
│
├───config
│       bank_data_preprocessing_config.json
│       config.json
│
├───data
│   ├───processed
│   └───raw
│           bank_data.csv
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
│   │   │   exceptions.py
│   │   │   utils.py
│   │
│   ├───data_preprocessing
│   │       data_cleaner.py
│   │       data_encoder.py
│   │       encoder_base.py
│   │       label_encoder.py
│   │       one_hot_encoder.py
│   │       ordinal_encoder.py
│   │
│   ├───eda
│   │       eda_visualizer.py
│   │
│   ├───models
│   │       data_splitter.py
│   │       model_evaluator.py
│   │       model_trainer.py
│   │
│   └───notebooks
│           original_churn_notebook.ipynb
│
└───tests
        churn_script_logging_and_tests.py
        test_data_cleaner.py
        test_data_encoder.py
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
Command line example, for cleaning a dataset :
```bash
python src/data_processing/data_cleaner.py --config=config/bank_data_cleaner_config.json --csv=data/bank_data.csv --result=data/cleaned_bank_data.csv
```
Command line example, for encoding a dataset :
```bash
python src/data_processing/data_encoder.py --config=config/bank_data_encoder_original_column_names_config.json --csv=data/bank_data.csv --result=data/encoded_bank_data.csv
```
See the python script `churn_library.py`, for an example of a complete ML pipeline.
For executing the ML pipeline on a sample dataset, using values set in config/config.json, run:
```bash
churn_library
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

