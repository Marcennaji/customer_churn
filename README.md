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
### 👅 **2. Setting Up a Virtual Environment (Recommended)**

To ensure a consistent and isolated development environment, it is recommended to use a **Python virtual environment** (`venv`). This prevents conflicts with global Python packages and ensures compatibility across different systems.

#### **Check Python Version**
This project is tested with **Python 3.10**. You can check your version by running:
```bash
python --version
```
If it’s not Python 3.10, install it from [python.org](https://www.python.org/downloads/) or using your system’s package manager. 

#### **Create and Activate a Virtual Environment**
##### **On macOS & Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```
##### **On Windows**
```powershell
py -3.10 -m venv venv
venv\Scripts\activate
```
Once activated, you should see `(venv)` in your terminal prompt, indicating that you are inside the virtual environment.
Verify that it uses python 3.10.
```bash
python --version
```

Using a virtual environment ensures a clean, conflict-free workspace for developing and running the project. 🚀

### 👅 **3. Install Dependencies**
After activating the virtual environment, install the required dependencies:
```bash
pip install -r requirements.txt
```
---

## 🚀 **ML pipeline usage**
### **1. Configure the pipeline**
- Open `config/config.json` and set your root directory:
```json
  "root_directory": "path/to/customer_churn"
  ``` 
- To enable **evaluation-only mode**, if you already have trained models in your `models` directory, set:
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

Instead of `autopep8` and `pylint`, this project uses `black` and `ruff` for code formatting and linting, as configured in `.pre-commit-config.yaml`. The reasons for this choice:
- **black**: Provides consistent, opinionated formatting, ensuring uniformity across all scripts.
- **ruff**: A fast Python linter and formatter, which also provides additional checks and optimizations beyond `black`.

However, for those who still want a pylint report, a dedicated script, `pylint_checker.py`, is provided as a convenience tool, to analyze all Python files recursively and generate both a summary and a detailed report.
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

