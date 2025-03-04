# Customer Churn Prediction (Work in Progress)

## Description
Udacity MLOPS engineer Nanodegree.
This project implements an MLOps pipeline for customer churn prediction. It includes data collection, preprocessing, model training, and evaluation.

## Project Architecture
```
customer_churn/
│── src/
│   ├── common/
│   │   ├── config_loader.py
│   │   ├── logger_config.py
│   │   ├── __init__.py
│   ├── data_processing/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── __init__.py
│   ├── models/
│   │   ├── model_train.py
│   │   ├── model_evaluate.py
│   │   ├── __init__.py
│   ├── churn_library.py
│── tests/
│── notebooks/
│── config/
│── README.md
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

