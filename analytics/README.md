# Module 2 - Analytics Pipeline

This module performs exploratory data analysis, data preprocessing, classification, regression, model evaluation, imbalance handling, hyperparameter tuning, and model persistence using the Titanic dataset.

## Files

- `01_eda.py` - Data profiling, cleaning, statistical analysis, and visualization
- `02_modeling.py` - Machine learning modeling and evaluation
- `titanic.csv` - Cleaned Titanic dataset used for modeling
- `titanic_survival_pipeline.joblib` - Saved complete survival prediction pipeline

## Exploratory Data Analysis

The EDA includes:

- Dataset shape and summary statistics
- Missing-value percentage analysis
- Missing-value handling
- Mean, median, and mode analysis
- IQR-based outlier detection for age and fare
- Correlation analysis
- Survival analysis by gender and passenger class
- Age distribution analysis
- Fare distribution analysis
- Standardization of age and fare

## Key Findings

- Female passengers had a much higher survival rate than male passengers.
- First-class passengers showed better survival outcomes.
- Age was slightly right-skewed.
- Fare was strongly right-skewed.
- Passenger class and fare showed a strong relationship.

## Classification Models

Three classification models were trained to predict passenger survival:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7989 | 0.7797 | 0.6667 | 0.7188 | 0.8194 |
| Decision Tree | 0.7709 | 0.7059 | 0.6957 | 0.7007 | 0.7514 |
| Random Forest | 0.7877 | 0.7460 | 0.6812 | 0.7121 | 0.8205 |

## Imbalance Handling

Random Forest was evaluated using three approaches:

- Baseline model
- Class Weight Balanced
- SMOTE

Results:

| Method | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Baseline | 0.7460 | 0.6812 | 0.7121 |
| Class Weight Balanced | 0.7500 | 0.7391 | 0.7445 |
| SMOTE | 0.6901 | 0.7101 | 0.7000 |

The class-weight-balanced approach achieved the best F1 score.

## Random Forest Hyperparameter Tuning

GridSearchCV was used to tune the Random Forest model.

Best parameters:

- `max_depth = 5`
- `max_features = sqrt`
- `n_estimators = 50`

Best CV F1 Score: **0.7504**

OOB Score: **0.809**

## Regression - Fare Prediction

Linear Regression was used to predict passenger fare.

Results:

- MAE: **25.4461**
- RMSE: **39.0466**
- R2 Score: **0.0147**
- Adjusted R2: **0.0092**

The low R2 score indicates that age alone explains very little of the variation in passenger fare.

## Final Recommendation

Logistic Regression is recommended as the final survival classification model because it provides a strong overall balance of accuracy, F1 score, and ROC-AUC.

## Model Persistence

The complete preprocessing and Logistic Regression pipeline was saved using Joblib as:

`titanic_survival_pipeline.joblib`

The saved pipeline was successfully reloaded and tested directly on raw input data.

## Conclusion

Module 2 successfully demonstrates the complete analytics workflow from exploratory data analysis and preprocessing to machine learning model development, evaluation, tuning, and model persistence.