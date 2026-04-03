## 📌 Project Overview
This project focuses on predicting customer churn for a telecommunications company using machine learning classification models. The goal is to  identify customers at risk of cancelling
their service, enabling the business to take a proactive  retention actions.
The project consists of two phases: **Exploratory Data Analysis (EDA)** to understand the data and uncover key churn patterns, and **Model Training** where three classification models were trained 
and compared: Logistic Regression, Random Forest, XGBoost - each evaluated on default settings and tuned using GridSearchCV with F1-score optimization.

## 📊 Dataset Information
The data used in this project comes from Telco Customer Churn dataset  available on Kaggle.
* **Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* **Records:** 7043 customers
* **Features:** 21 columns (20 features + 1 target variable)
* **Class imbalance:** Churn = No: 5174 (73.58 %) | Churn = Yes: 1869 (26.42 %)
> Column descriptions sourced from the official Kaggle dataset page.

## 🛠️ Tech Stack & Tools
* **Languages:** Python
* **Data Manipulation:** `pandas`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Environment:** Jupyter Lab


## 📂 Project Architecture
The project is structured into 2 logical stages (Jupyter Notebooks).
### 1️⃣ Phase 1: Data Cleaning & EDA
File: `01_Data_Cleaning_and_EDA.ipynb`
* **Data Quality Check:** Design a custom class to identify missing values and duplicates. Handle missing values.
* **Univariate Analysis:** Analysis data to better understand distribution.
* **Bivariate  Analysis:** Relationships between features and churn target variable
* **Correlation Analysis:** Investigating relationships between numerical features and churn to select independent variables for modeling

### 2️⃣ Phase 2: Classification Models Training
File: `02_Classification_Models_Training.ipynb`
* **Preprocessing:** Label encoding of categorical variables, train/test split with stratification. Class imbalance (73/27) was addressed via StratifiedKFold cross-validation and class_weight='balanced' during GridSearchCV tuning.
* **Logistic Regression:** Baseline model + GridSearchCV tuning (F1 optimization)
* **Random Forest Classifier:** Baseline model + GridSearchCV tuning (F1 optimization)
* **XBGClassifier:** Baseline model + GridSearchCV tuning (F1 optimization)
* **Feature Importance:** Analysis of key churn predictors for each tuned model
* **ROC-AUC:** Comparison of all tuned models

## 📈 Results

|Model|Precision|Recall|Specificity|Accuracy|F1|ROC-AUC|FN|
|:-----|:------|:------|:--------|:-------|:------|:-----|:-----|
|LR Default|67.23 %|55.08 %|90.34 %|80.98 %|60.59 %|-|168|
|LR Tuned|54.12 %|82.26 %|74.69 %|76.79 %|65.40 %|85.43 %|65|
|RF Default|66.91 %|48.66 %|91.30%|79.99%|56.35 %|-|192|
|RF Tuned|58.91 %|75.13 %|81.06 %|79.49 %|66.04 %|85.57 %|93|
|XGB Default|63.14 %|52.67 %|88.89 %|79.28 %|57.43 %|-|177|
|XGB Tuned|54.58 %|81.28 %|75.56 %|77.08 %|65.31 %|86.27 %|70|

* 🏆**Best Model: Logistic Regression Tuned**
* Highest Recall (82.26 %)
* Lowest missed churners (FN = 65)
* Comparable F1 and Precision to more complex models
* Simple, interpretable and easy to deploy

![ROC Curve](images/roc_curve.png)

* All three tuned models achieve nearly identical ROC-AUC score (LR: 0.854, RF: 0.857, XGB: 0.863), indicating comparable discriminative ability. The final model selection was theraforce basen on Recall and FN - metrics that better reflect the business
objective of minimizing missed churners.

## 🔍 Key Findings
* **Contract type** is the strongest churn predictor across all models
* Customers on **month-to-month contracts** have a ~ 43 % churn rate vs ~ 3 % for **two-years contracts**
* **New customers** (0-2 months tenure) are the most likely to churn
* Customers with **high monthly charges (70-100 USD)** are at significantly greater risk
* Customers without **OnlineSecurity** or **TechSupport** are more prone to cancellation (XGBClassifier finding)



## 🚀 How to Run (Local Setup)

To reproduce the analysis on your local machine, follow these steps:

**1. Clone the repository and navigate to the project directory:**
```bash
git clone https://github.com/Oliwer1992/telco-customer-churn.git
cd telco-customer-churn
```

**2. Set up a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Jupyter Notebooks:**
Launch Jupyter Lab or Jupyter Notebook and execute the files in the following sequential order:
```bash
jupyter lab
```
* `01_Data_Cleaning_and_EDA.ipynb`  (Cleans data and generates visualizations)
* `02_Classification_Models_Training.ipynb` (Train models and evaluates performance)

> **Note:** Ensure that the raw CSV data files are placed in the correct `data/` directory.
