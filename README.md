# 🤖 Kaggle Intermediate Machine Learning

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23FF6600.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-%2320BEFF.svg?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/learn/intermediate-machine-learning)

A structured collection of notebooks implementing **intermediate-level machine learning techniques** from Kaggle's *Intermediate Machine Learning* course. This repository goes beyond basic modeling to cover the practical skills needed to build **robust, production-ready ML pipelines** — including missing value handling, categorical encoding, cross-validation, XGBoost, and preventing data leakage.

---

## 📂 Project Structure

```
Kaggle-Intermediate-ML/
│
├── exercise-introduction.ipynb            # Step 1: Course intro & baseline model
├── exercise-missing-values.ipynb          # Step 2: Imputation strategies
├── exercise-categorical-variables.ipynb   # Step 3: Encoding categorical features
├── exercise-pipelines.ipynb               # Step 4: Scikit-learn Pipelines
├── exercise-cross-validation.ipynb        # Step 5: K-Fold cross-validation
├── exercise-xgboost.ipynb                 # Step 6: XGBoost gradient boosting
└── exercise-data-leakage.ipynb            # Step 7: Detecting & preventing leakage
```

---

## 🗂️ Datasets Used

| Dataset | Used In |
|---|---|
| **Melbourne Housing Dataset** | Missing values, categorical encoding, pipelines |
| **Ames Housing Dataset** | Cross-validation, XGBoost |
| **Credit Card Dataset** | Data leakage detection |

---

## 📓 Notebook Walkthrough

### Step 1 — Introduction
**Notebook:** `exercise-introduction.ipynb`

- Reviewed the supervised ML workflow: load data → select features → train model → evaluate
- Established a baseline Random Forest model on the Melbourne Housing dataset
- Set the foundation for all techniques covered in subsequent steps

---

### Step 2 — Handling Missing Values
**Notebook:** `exercise-missing-values.ipynb`

Three strategies were compared side by side:

| Strategy | Description |
|---|---|
| **Drop columns** | Remove any column with missing values |
| **Simple Imputation** | Fill missing values with mean/median |
| **Extension Imputation** | Impute + add a boolean column indicating which rows were imputed |

- Evaluated each strategy using MAE on a validation set
- Discovered that extension imputation often outperforms simple imputation by letting the model learn that missingness itself is informative

---

### Step 3 — Categorical Variables
**Notebook:** `exercise-categorical-variables.ipynb`

Three encoding approaches were implemented and compared:

| Encoding | Description | Best For |
|---|---|---|
| **Drop Categorical** | Remove all non-numeric columns | Quick baseline |
| **Ordinal Encoding** | Assign integer rankings to categories | High-cardinality features |
| **One-Hot Encoding** | Create binary columns per category | Low-cardinality features |

- Applied encoding strategies to real housing data features
- Evaluated MAE for each approach to determine the best fit for the dataset

---

### Step 4 — Pipelines
**Notebook:** `exercise-pipelines.ipynb`

- Built a full **scikit-learn Pipeline** bundling preprocessing and modeling into a single object
- Pipeline structure:
  1. **Numerical features** → `SimpleImputer` (mean strategy)
  2. **Categorical features** → `SimpleImputer` (most frequent) + `OneHotEncoder`
  3. **Model** → `RandomForestRegressor`
- Used `ColumnTransformer` to apply different preprocessing to different column types simultaneously
- Key benefit: Pipelines prevent data leakage by ensuring preprocessing is fitted only on training data and applied consistently to validation/test data

---

### Step 5 — Cross-Validation
**Notebook:** `exercise-cross-validation.ipynb`

- Replaced single train/validation split with **K-Fold Cross-Validation** (5 folds)
- Used `cross_val_score` with `neg_mean_absolute_error` scoring
- Compared cross-validated MAE scores across different pipeline configurations
- Key insight: Cross-validation gives a more reliable estimate of model performance by using all data for both training and validation across different folds

---

### Step 6 — XGBoost
**Notebook:** `exercise-xgboost.ipynb`

- Implemented **XGBoost (Extreme Gradient Boosting)** — a state-of-the-art algorithm for tabular data
- Tuned key hyperparameters:

| Parameter | Effect |
|---|---|
| `n_estimators` | Number of boosting rounds — more trees = better fit but slower |
| `learning_rate` | Step size per round — lower = more robust but needs more trees |
| `early_stopping_rounds` | Stops training when validation score stops improving |
| `max_depth` | Controls tree complexity to avoid overfitting |

- Used `eval_set` to monitor validation MAE during training
- XGBoost significantly outperformed the baseline Random Forest model

---

### Step 7 — Data Leakage
**Notebook:** `exercise-data-leakage.ipynb`

Two critical types of data leakage were studied and detected:

| Leakage Type | Description | Risk |
|---|---|---|
| **Target Leakage** | Features that are only known *after* the target is determined | Model looks artificially accurate but fails in production |
| **Train-Test Contamination** | Preprocessing fitted on the full dataset before splitting | Validation scores are overly optimistic |

- Analyzed a real credit card dataset to identify features causing target leakage
- Understood why pipelines are the correct solution to prevent train-test contamination

---

## 📊 Techniques Summary

| Technique | Tool Used |
|---|---|
| Missing value imputation | `sklearn.impute.SimpleImputer` |
| Ordinal encoding | `sklearn.preprocessing.OrdinalEncoder` |
| One-hot encoding | `sklearn.preprocessing.OneHotEncoder` |
| Column-wise preprocessing | `sklearn.compose.ColumnTransformer` |
| Full ML pipeline | `sklearn.pipeline.Pipeline` |
| K-Fold cross-validation | `sklearn.model_selection.cross_val_score` |
| Gradient boosting | `xgboost.XGBRegressor` |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Jupyter Notebook / Kaggle Notebooks
- **Libraries:**
  - `pandas` — Data loading and manipulation
  - `scikit-learn` — Preprocessing, pipelines, cross-validation
  - `xgboost` — Gradient boosting model

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/malaikaahsan/Kaggle-Intermediate-ML.git
cd Kaggle-Intermediate-ML
```

2. Install dependencies:
```bash
pip install pandas scikit-learn xgboost
```

3. Download the datasets:
   - [Melbourne Housing Dataset](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)
   - [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

4. Run notebooks in order (Step 1 → Step 7)

> **Note:** These notebooks were originally developed on Kaggle. Dataset file paths may need updating if running locally.

---

## 💡 Key Takeaways

- **Pipelines** are not just convenient — they are essential for preventing data leakage in real projects
- **Cross-validation** gives a far more honest picture of model performance than a single train/test split
- **XGBoost** is one of the most powerful algorithms for structured/tabular data and is widely used in industry
- **Data leakage** is one of the most dangerous and common mistakes in ML — models with leakage look great during development but fail completely in production
- The combination of Pipelines + Cross-Validation + XGBoost is a professional-grade ML workflow used in real-world data science

---

## 👩‍💻 Author

**Malaika Ahsan**
BS Computer Science — PUCIT, Lahore
[GitHub](https://github.com/malaikaahsan)
