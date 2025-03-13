# Customer Churn Prediction

## 📌 Description
This project is part of the **Udacity MLOps Engineer Nanodegree – February 2025**.

The goal is to **refactor** the existing notebook (`src/notebooks/churn_notebook.ipynb`) by applying **clean code best practices** and improving modularity.  
For implementation details, refer to `src/notebooks/Guide.jpynb`.

### 🔹 **Key Features**
✔ **PEP 8 compliant & well-documented**  
✔ **Modular, maintainable, and reusable ML pipeline**  
✔ **Flexible configuration via `config/config.json`**  
✔ **Stores EDA plots, trained models, and evaluation metrics**  
✔ **Efficient handling of categorical variables**  

---

## 📂 **Project Structure**
## Project Structure

```
customer_churn/                     Main project directory
│── README.md                       Project documentation
│── requirements.txt                Dependencies
│── config/                         Configuration files
│   ├── config.json                 Main configuration file
│   ├── preprocessing_config.json
│   ├── data_splitting_profiles.json
│   ├── training_config.json
│── data/                           Dataset storage
│   ├── raw/                        Original data
│   │   ├── bank_data.csv
│   ├── processed/                  Preprocessed data
│── logs/                           Logging output
│   ├── customer_churn.log
│── models/                         Trained models
│   ├── LogisticRegression.pkl
│   ├── RandomForestClassifier.pkl
│── results/                        Evaluation results
│   ├── images/                     Plots and visualizations
│   │   ├── eda/                    Exploratory Data Analysis
│   ├── json/                       Evaluation metrics
│── src/                            Source code
│   ├── churn_library.py            Main ML pipeline
│   ├── config_manager.py           Configuration handler
│   ├── logger_config.py            Logging setup
│   ├── common/                     Shared utilities
│   │   ├── exceptions.py           Custom exceptions
│   ├── data_preprocessing/         Data cleaning & encoding
│   │   ├── data_cleaner.py
│   │   ├── data_encoder.py
│   ├── eda/                        Exploratory Data Analysis
│   │   ├── eda_visualizer.py
│   ├── models/                     Model training & evaluation
│   │   ├── data_splitter.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluator.py
│   ├── notebooks/                  Jupyter notebooks, provided as reference
│   │   ├── Guide.ipynb
│   │   ├── churn_notebook.ipynb
│── tests/                          Unit tests using pytest
│   ├── test_data_cleaner.py
│   ├── test_data_encoder.py
│   ├── test_data_splitter.py
│   ├── test_data_visualizer.py
│   ├── test_model_trainer.py
│   ├── test_model_evaluator.py
│── utils/                          Utility scripts
│   ├── pylint_checker.py           automatically generates a pylint report for all source code

```
---

## ⚙ **Installation**
### 👅 **1. Clone the repository**
```bash
git clone https://github.com/Marcennaji/customer_churn.git
cd customer_churn
```
### 📦 **2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 **Usage**
### **1. Configure the pipeline**
- Open `config/config.json` in the root directory  
- Set the `"data_path"` to your dataset location  
- To enable **evaluation-only mode**, set:
  ```json
  "eval_only": true
  ```

### **2. Run the ML pipeline**
```bash
python src/churn_library.py --config=config/config.json
```

---

## 🛠 **Running Tests**
### **Run all unit tests**
```bash
pytest
```
### **Run tests with verbose output**
```bash
pytest -v
```
- All test files are located in the `tests/` directory  
- Test configuration is managed via `pytest.ini`  

---

## 📍 **Logging & Debugging**
### **Check Logs**
```bash
cat logs/customer_churn.log
```
- Logs include **both info and error messages**
- Stored in `customer_churn.log` after script execution.

---

## 🔍 **Code formatting and linting**

To ensure high-quality and maintainable code, this project follows strict formatting and linting guidelines.

### Formatting

Instead of `autopep8` and `pylint`, this project uses `black` and `ruff` for code formatting and linting, as configured in `.pre-commit-config.yaml`. The reasons for this choice:

- **black**: Provides consistent, opinionated formatting, ensuring uniformity across all scripts.
- **ruff**: A fast Python linter and formatter, which also provides additional checks and optimizations beyond `black`.

#### autopep8 vs. black

Both autopep8 and black format Python code, but:
autopep8: Focuses on fixing PEP 8 violations (less opinionated).
black: Enforces a strict, opinionated style (makes all code look uniform).
black is the better choice for standardization and team projects.

#### pylint vs. ruff

Both pylint and ruff check for code style and quality, but:
pylint: More thorough but slow and sometimes overly strict.
ruff: Faster, supports many pylint rules, and can auto-fix some issues.
ruff is a modern alternative to pylint, often preferred for speed.

However, for those who want a pylint report, a dedicated script, `pylint_checker.py`, is provided to analyze all Python files recursively and generate both a summary and a detailed report.
Simply run:

```bash
python utils/pylint_checker.py
```
---

## 📊 **Stored Images & Models**
### ✅ **EDA Plots (Saved in `results/images/eda`)**
✔ **Univariate (quantitative)** → `histogram_age.png`  
✔ **Univariate (categorical)** → `bar_chart_marital_status.png`  
✔ **Bivariate Plot** → `correlation_heatmap.png`

### ✅ **Evaluation Plots (Saved in `results/images/`)**
✔ **ROC Curves** → `roc_curve.png`  
✔ **Feature Importances** → `feature_importance_RandomForestClassifier.png`  
✔ **SHAP Explanation** → `shap_RandomForestClassifier.png`

### ✅ **Stored Models (`models/` Directory)**
✔ **Logistic Regression (`LogisticRegression.pkl`)**  
✔ **Random Forest (`RandomForestClassifier.pkl`)**  

Models are stored in **.pkl format using Joblib** for easy deployment.

---

## 🔬 **Handling Categorical Variables**
- Uses **mean encoding** or **one-hot encoding** based on configuration  
- Efficiently processes categorical columns via **looping**  
- Defined in `src/data_preprocessing/data_encoder.py`

---

## 🛠 **Future Improvements**
✔ **Optimize hyperparameter tuning strategy**  
✔ **Enhance model versioning with MLflow**  
✔ **Add Docker support for easy deployment**  

---

## 👨‍💼 **Author**
**Marc Ennaji**  
🌍 [GitHub](https://github.com/Marcennaji) | 📝 [LinkedIn](https://linkedin.com/in/marcennaji)  

---

## 🐝 **License**
This project is licensed under the **MIT License**.  

