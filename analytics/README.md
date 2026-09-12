# Module 2 - Analytics Pipeline

This module performs exploratory data analysis, preprocessing, classification, imbalance handling, hyperparameter tuning, regression, model comparison, and model persistence using the Titanic dataset.

## Files

- `01_eda.py` - Data profiling, cleaning, statistical analysis, visualization, correlation analysis, and standardization checks
- `02_modeling.py` - Classification, imbalance handling, hyperparameter tuning, regression, evaluation, and model persistence
- `titanic.csv` - Offline Titanic dataset used by the modeling pipeline
- `titanic_survival_pipeline.joblib` - Saved complete survival prediction pipeline

## How to Run

From the project root, activate the virtual environment and run:

```bash
python analytics/01_eda.py
```

After completing the EDA, run:

```bash
python analytics/02_modeling.py
```

The scripts print the analysis and model results in the terminal and display the required visualizations.

## Part A - Exploratory Data Analysis

The EDA includes:

- Dataset shape, structure, and summary statistics
- Missing-value percentage calculation
- Missing-value handling based on percentage thresholds
- Histogram and boxplot analysis for age and fare
- IQR-based outlier detection
- Fare mean, median, and mode analysis
- Survival-rate analysis by sex
- Survival-rate analysis by passenger class
- Survival-rate analysis by sex and passenger class
- Six-variable correlation matrix
- Correlation heatmap
- Identification of the two strongest off-diagonal correlations
- Four multivariate visualizations with written interpretations
- Standardization sanity checks for age and fare

## Missing-Value Strategy

Missing values are handled according to the percentage of missing data:

- Less than 5%: affected rows are removed
- 5% to 30%: values are imputed
- More than 30%: the column may be removed when justified

For the Titanic dataset:

- `age` is imputed using the median
- rows with missing `embarked` and `embark_town` values are removed
- `deck` is removed because a large percentage of its values are missing

## Key EDA Findings

- Female passengers show substantially higher survival rates than male passengers.
- Passenger class is associated with survival outcomes.
- First-class passengers generally show better survival outcomes than third-class passengers.
- Fare has a right-skewed distribution and contains several high-value outliers.
- Passenger class and fare show an important relationship.
- Age alone does not clearly separate survivors from non-survivors.

## Standardization Check

`age` and `fare` are standardized during the EDA sanity check.

After standardization:

- the mean is approximately 0
- the standard deviation is approximately 1

This EDA scaling is separate from model preprocessing. The modeling pipeline learns preprocessing parameters from the training data only.

## Part B - Predictive Modeling

The classification target is:

`survived`

A stratified train-test split is used so that the survival-class proportions remain approximately consistent between training and testing data.

Stratification is important because the Titanic survival classes are not perfectly balanced. It helps maintain approximately the same survived/not-survived proportions in both the training and testing sets.

## Train-Only Preprocessing

The modeling pipeline uses:

- median imputation for numeric features
- most-frequent imputation for categorical features
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical features

Preprocessing is fitted using the training data only.

The held-out test data is transformed using parameters learned from the training data. This avoids preprocessing leakage from the test set.

## Classification Models

Three classifiers are trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Each model is evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The Decision Tree is also visualized using the transformed feature names and survival class names.

## Final Model Comparison

The three classification models were evaluated on the same held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.8268 | 0.7969 | 0.7391 | 0.7669 | 0.7994 |
| Random Forest | 0.7989 | 0.7797 | 0.6667 | 0.7188 | 0.8322 |

Decision Tree achieved the highest F1 score of **0.7669** and the highest accuracy of **0.8268** among the three classifiers.

## Imbalance Handling

Random Forest was evaluated using three approaches:

1. Baseline Random Forest
2. Random Forest with `class_weight="balanced"`
3. Random Forest with SMOTE

Results:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline | 0.7797 | 0.6667 | 0.7188 |
| Class Weight Balanced | 0.7463 | 0.7246 | 0.7353 |
| SMOTE | 0.7500 | 0.7391 | 0.7445 |

SMOTE produced the highest F1 score of **0.7445** among the three imbalance-handling approaches.

SMOTE was applied only to the training data. No synthetic samples were created for the test set, which prevents test-data leakage.

## Random Forest Hyperparameter Tuning

`GridSearchCV` was used to tune:

- `n_estimators`
- `max_depth`
- `max_features`

Five-fold cross-validation was used with F1 score as the optimization metric.

Best parameters:

- `max_depth = 10`
- `max_features = sqrt`
- `n_estimators = 50`

Best Cross-Validation F1 Score: **0.7487**

OOB Score: **0.8230**

## Regression - Fare Prediction

A multivariate Linear Regression model was used to predict passenger `fare` using three features:

- `age`
- `pclass`
- `sibsp`

The final regression results were:

| Metric | Value |
|---|---:|
| MAE | 19.0530 |
| RMSE | 31.2499 |
| R² | 0.3689 |
| Adjusted R² | 0.3581 |

The R² value indicates that the selected features explain approximately **36.89%** of the variation in passenger fare.

A residual plot was generated to examine the relationship between predicted fare and residual errors. The residual spread changes noticeably across prediction levels, suggesting possible **heteroscedasticity**.

Regression metrics are reported separately from classification metrics because classification and regression represent different prediction tasks and their metrics are not directly comparable.

## Final Recommendation

Decision Tree is recommended as the final survival classifier in this experiment because it achieved the highest held-out **F1 score of 0.7669** and the highest **accuracy of 0.8268** among the three tested classifiers.

Its recall of **0.7391** was also higher than Logistic Regression and Random Forest in this run.

Logistic Regression achieved the highest ROC-AUC of **0.8437**, showing strong ranking ability, but its F1 score was lower at **0.7244**.

Considering accuracy, precision, recall, and F1 together, Decision Tree provides the strongest overall performance for this experiment.

## Model Persistence

The selected complete classification pipeline is saved using Joblib as:

`titanic_survival_pipeline.joblib`

The saved artifact contains:

- preprocessing
- numeric imputation
- categorical imputation
- numeric scaling
- categorical encoding
- trained classifier

The saved pipeline was successfully reloaded using `joblib.load()` and tested directly on raw, unprocessed test rows.

This confirms that preprocessing and prediction are packaged together in the saved pipeline.

## Conclusion

Module 2 demonstrates a complete analytics and machine-learning workflow covering exploratory data analysis, missing-value treatment, statistical analysis, visualization, leakage-safe preprocessing, classification, imbalance handling, hyperparameter tuning, multivariate regression, model evaluation, final model recommendation, and model persistence.