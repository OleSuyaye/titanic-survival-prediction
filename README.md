# Titanic Survival Prediction

A machine learning project predicting passenger survival on the Titanic, built for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition.

**Leaderboard score:** 0.76315 (accuracy)

## Overview

This project walks through a full classification workflow predicting whether a passenger survived the Titanic disaster based on features like class, sex, age, and fare.

## Dataset

- Source: Kaggle Titanic competition (`train.csv`)
- 891 passengers, 12 original columns
- Target column: `Survived` (0 = did not survive, 1 = survived)
- Key features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

### Missing values

| Column | % Missing |
|---|---|
| Cabin | 77.1% |
| Age | 19.9% |
| Embarked | 0.2% |

`Cabin` was dropped due to excessive missingness. `Age` and `Embarked` were handled via imputation in the preprocessing pipeline.

## Workflow

1. **Data cleaning** — duplicate checks, missing value analysis, dropped `Cabin`
2. **EDA** — target distribution, feature types (numerical vs categorical)
3. **Preprocessing pipeline**
   - Numerical features (`Age`, `SibSp`, `Parch`, `Fare`): median imputation + standard scaling
   - Categorical features (`Pclass`, `Sex`, `Embarked`): most-frequent imputation + one-hot encoding
   - Combined via `ColumnTransformer`
4. **Model comparison** — 5 candidate models evaluated with 5-fold stratified cross-validation (F1 and ROC-AUC):
   - Logistic Regression
   - K-Nearest Neighbors
   - Random Forest
   - Support Vector Machine (SVM)
   - Gradient Boosting
5. **Hyperparameter tuning** on the best-performing model (Random Forest), comparing three search strategies:
   - Grid Search CV
   - Randomized Search CV
   - Optuna (Bayesian optimization)
6. **Final model training and Kaggle submission**

## Model Comparison Results (5-fold CV)

| Model | Mean F1 | Mean ROC-AUC |
|---|---|---|
| Logistic Regression | 0.730 | — |
| K-Nearest Neighbors | 0.719 | — |
| Random Forest | 0.735 | — |
| SVM | 0.745 | — |
| Gradient Boosting | 0.705 | — |

## Hyperparameter Tuning Results

| Method | Best F1 | n_estimators | max_depth | min_samples_split | min_samples_leaf |
|---|---|---|---|---|---|
| Grid Search | 0.7570 | 200 | 10 | 2 | 1 |
| Randomized Search | 0.7582 | 100 | 10 | 5 | 1 |
| Optuna | 0.7561 | 501 | 16 | 6 | 1 |

All three methods converged to comparable performance, so the simpler, faster configuration (Randomized Search result) was preferred for the final model.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (pipelines, preprocessing, model selection)
- XGBoost
- Optuna
- matplotlib, seaborn

## How to Run

```bash
git clone <your-repo-url>
cd titanic-survival-prediction
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
jupyter notebook
```

## Future Improvements

- Feature engineering (title extraction from `Name`, family size from `SibSp` + `Parch`, cabin deck from partial `Cabin` data)
- Ensemble/stacking of top-performing models
- Deeper exploration of `XGBClassifier`, which was imported but not yet included in the model comparison

## Author

Kevin Ng'ang'a
