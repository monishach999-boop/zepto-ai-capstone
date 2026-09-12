# =========================================================
# MODULE 2 - PART B: PREDICTIVE MODELING
# =========================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "titanic.csv"
MODEL_PATH = BASE_DIR / "titanic_survival_pipeline.joblib"


# =========================================================
# TASK 7: LOAD SAME TITANIC DATA + STRATIFIED SPLIT
# =========================================================

print("\n========== TASK 7: STRATIFIED TRAIN-TEST SPLIT ==========")

# Load the committed offline Titanic dataset.
# No second sns.load_dataset() call is used here.
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# Classification target
y = df["survived"]

# Features
X = df.drop(columns=["survived"])


print("\nOverall class distribution:")
print(y.value_counts())

print("\nOverall class proportions:")
print(y.value_counts(normalize=True).round(4))


# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


print("\nTraining class proportions:")
print(y_train.value_counts(normalize=True).round(4))


print("\nTesting class proportions:")
print(y_test.value_counts(normalize=True).round(4))


print("\nStratification Justification:")
print(
    "The Titanic target classes are not perfectly balanced. "
    "A stratified split preserves approximately the same "
    "survived/not-survived proportion in both training and "
    "testing sets, making model evaluation more reliable."
)


# =========================================================
# TASK 8: TRAIN-ONLY PREPROCESSING
# =========================================================

print("\n========== TASK 8: PREPROCESSING PIPELINE ==========")


# Selected useful features
numeric_features = [
    "age",
    "pclass",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


selected_features = numeric_features + categorical_features

X_train_model = X_train[selected_features].copy()
X_test_model = X_test[selected_features].copy()


# Numeric pipeline
numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical pipeline
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combined transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


print(
    "Preprocessing includes median imputation, "
    "categorical imputation, one-hot encoding, "
    "and numeric standardization."
)

print(
    "The preprocessing pipeline is fitted only "
    "when model.fit() is called on X_train."
)

print(
    "The test data is transformed using the "
    "parameters learned from training data only."
)


# =========================================================
# TASK 9: TRAIN THREE CLASSIFIERS
# =========================================================

print("\n========== TASK 9: TRAIN THREE CLASSIFIERS ==========")


logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]
)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}


for name, model in models.items():

    model.fit(
        X_train_model,
        y_train
    )

    print(f"{name} trained successfully.")


# =========================================================
# TASK 9 EXTRA: DECISION TREE VISUALIZATION
# =========================================================

print("\nCreating Decision Tree visualization...")


tree_preprocessor = decision_tree_model.named_steps[
    "preprocessor"
]

tree_classifier = decision_tree_model.named_steps[
    "classifier"
]


feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)


plt.figure(figsize=(22, 12))

plot_tree(
    tree_classifier,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title(
    "Decision Tree - Titanic Survival"
)

plt.tight_layout()
plt.show()


# =========================================================
# TASK 10: EVALUATE ALL THREE CLASSIFIERS
# =========================================================

print("\n========== TASK 10: MODEL EVALUATION ==========")


classification_rows = []


for name, model in models.items():

    y_pred = model.predict(
        X_test_model
    )

    y_probability = model.predict_proba(
        X_test_model
    )[:, 1]


    cm = confusion_matrix(
        y_test,
        y_pred
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )


    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("\nConfusion Matrix:")
    print(cm)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )


    classification_rows.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": auc
        }
    )


classification_results = pd.DataFrame(
    classification_rows
)


# =========================================================
# TASK 11: IMBALANCE HANDLING COMPARISON
# =========================================================

print(
    "\n========== TASK 11: IMBALANCE HANDLING =========="
)


# IMPORTANT:
# fit_transform happens ONLY on the training split.
imbalance_preprocessor = preprocessor


X_train_ready = (
    imbalance_preprocessor
    .fit_transform(X_train_model)
)


# Test set is transform-only.
X_test_ready = (
    imbalance_preprocessor
    .transform(X_test_model)
)


# ---------------------------------------------------------
# 1. Baseline
# ---------------------------------------------------------

baseline_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


baseline_model.fit(
    X_train_ready,
    y_train
)


baseline_pred = baseline_model.predict(
    X_test_ready
)


# ---------------------------------------------------------
# 2. class_weight='balanced'
# ---------------------------------------------------------

balanced_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)


balanced_model.fit(
    X_train_ready,
    y_train
)


balanced_pred = balanced_model.predict(
    X_test_ready
)


# ---------------------------------------------------------
# 3. SMOTE - TRAINING DATA ONLY
# ---------------------------------------------------------

smote = SMOTE(
    random_state=42
)


X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_ready,
        y_train
    )
)


smote_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


smote_model.fit(
    X_train_smote,
    y_train_smote
)


smote_pred = smote_model.predict(
    X_test_ready
)


def imbalance_scores(
    name,
    predictions
):

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    return {
        "Method": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


imbalance_results = pd.DataFrame(
    [
        imbalance_scores(
            "Baseline",
            baseline_pred
        ),

        imbalance_scores(
            "Class Weight Balanced",
            balanced_pred
        ),

        imbalance_scores(
            "SMOTE",
            smote_pred
        )
    ]
)


print(
    "\nImbalance Handling Comparison:"
)

print(
    imbalance_results.round(4)
)


best_imbalance_method = (
    imbalance_results
    .sort_values(
        by="F1",
        ascending=False
    )
    .iloc[0]
)


print("\nImbalance Handling Conclusion:")

print(
    f"{best_imbalance_method['Method']} "
    "produced the highest F1 score "
    f"({best_imbalance_method['F1']:.4f}) "
    "among the three approaches."
)

print(
    "SMOTE was applied only to the training data, "
    "so no synthetic test samples were created and "
    "test-set information did not leak into training."
)


# =========================================================
# TASK 12: GRIDSEARCHCV + OOB SCORE
# =========================================================

print(
    "\n========== TASK 12: RANDOM FOREST TUNING =========="
)


param_grid = {

    "n_estimators": [
        50,
        100,
        200
    ],

    "max_depth": [
        None,
        5,
        10
    ],

    "max_features": [
        "sqrt",
        "log2"
    ]
}


rf_for_grid = RandomForestClassifier(
    random_state=42,
    oob_score=True
)


grid_search = GridSearchCV(
    estimator=rf_for_grid,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


grid_search.fit(
    X_train_ready,
    y_train
)


best_rf = grid_search.best_estimator_


print(
    "\nBest Parameters:"
)

print(
    grid_search.best_params_
)


print(
    "\nBest Cross-Validation F1:",
    round(
        grid_search.best_score_,
        4
    )
)


print(
    "OOB Score:",
    round(
        best_rf.oob_score_,
        4
    )
)


# =========================================================
# TASK 13: REGRESSION SIDE-TASK
# Predict FARE using THREE other features
# =========================================================

print(
    "\n========== TASK 13: REGRESSION - PREDICT FARE =========="
)


# Three predictor features
regression_features = [
    "age",
    "pclass",
    "sibsp"
]


X_reg = df[
    regression_features
].copy()


y_reg = df[
    "fare"
].copy()


X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)


# Regression preprocessing
regression_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


# ---------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)


r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


n = len(y_reg_test)

p = len(
    regression_features
)


adjusted_r2 = (
    1
    -
    (
        (1 - r2)
        * (n - 1)
        /
        (n - p - 1)
    )
)


print(
    f"\nMAE: {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R2: {r2:.4f}"
)

print(
    f"Adjusted R2: {adjusted_r2:.4f}"
)


# ---------------------------------------------------------
# Residual plot
# ---------------------------------------------------------

residuals = (
    y_reg_test
    -
    y_reg_pred
)


plt.figure(
    figsize=(8, 5)
)


plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.7
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - Fare Prediction"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Simple numerical heteroscedasticity check
# ---------------------------------------------------------

median_prediction = np.median(
    y_reg_pred
)


low_group = residuals[
    y_reg_pred <= median_prediction
]


high_group = residuals[
    y_reg_pred > median_prediction
]


low_spread = np.std(
    low_group
)

high_spread = np.std(
    high_group
)


spread_ratio = (
    max(
        low_spread,
        high_spread
    )
    /
    max(
        min(
            low_spread,
            high_spread
        ),
        1e-8
    )
)


print(
    "\nResidual Spread - Lower Predictions:",
    round(
        low_spread,
        4
    )
)


print(
    "Residual Spread - Higher Predictions:",
    round(
        high_spread,
        4
    )
)


print("\nHeteroscedasticity Conclusion:")


if spread_ratio >= 1.5:

    print(
        "The residual spread changes noticeably "
        "across prediction levels. This suggests "
        "possible heteroscedasticity."
    )

else:

    print(
        "The residual spread is reasonably similar "
        "across prediction levels. Strong "
        "heteroscedasticity is not clearly indicated."
    )


# Store regression results
regression_results = pd.DataFrame(
    {
        "Model": [
            "Linear Regression"
        ],

        "MAE": [
            mae
        ],

        "RMSE": [
            rmse
        ],

        "R2": [
            r2
        ],

        "Adjusted R2": [
            adjusted_r2
        ]
    }
)


# =========================================================
# TASK 14: FINAL MODEL COMPARISON
# =========================================================

print(
    "\n========== TASK 14: FINAL MODEL COMPARISON =========="
)


print(
    "\nCLASSIFICATION METRICS"
)

print(
    classification_results
    .round(4)
    .to_string(index=False)
)


print(
    "\nREGRESSION METRICS"
)

print(
    regression_results
    .round(4)
    .to_string(index=False)
)


print(
    "\nNote:"
)

print(
    "Classification and regression metrics are "
    "shown separately because they measure "
    "different types of prediction tasks and "
    "are not directly comparable on one scale."
)


# Find strongest classifier by F1
best_classifier_row = (
    classification_results
    .sort_values(
        by="F1",
        ascending=False
    )
    .iloc[0]
)


best_classifier_name = (
    best_classifier_row[
        "Model"
    ]
)


print(
    "\n========== FINAL WRITTEN RECOMMENDATION =========="
)


print(
    f"{best_classifier_name} achieved the highest "
    f"F1 score of {best_classifier_row['F1']:.4f} "
    "on the held-out test set."
)

print(
    f"It achieved accuracy "
    f"{best_classifier_row['Accuracy']:.4f} "
    f"and ROC-AUC "
    f"{best_classifier_row['ROC-AUC']:.4f}."
)

print(
    "The final classifier should balance overall "
    "accuracy with precision, recall and F1 rather "
    "than relying on accuracy alone."
)

print(
    f"Based on this run, {best_classifier_name} "
    "is recommended as the strongest of the three "
    "tested classifiers."
)


# =========================================================
# TASK 15: SAVE COMPLETE BEST PIPELINE
# =========================================================

print(
    "\n========== TASK 15: SAVE COMPLETE PIPELINE =========="
)


# Choose the fitted complete pipeline
if best_classifier_name == "Logistic Regression":

    final_pipeline = logistic_model


elif best_classifier_name == "Decision Tree":

    final_pipeline = decision_tree_model


else:

    final_pipeline = random_forest_model


# Save preprocessing + classifier together
joblib.dump(
    final_pipeline,
    MODEL_PATH
)


print(
    "\nComplete fitted pipeline saved to:"
)

print(
    MODEL_PATH
)


# =========================================================
# RELOAD PIPELINE
# =========================================================

loaded_pipeline = joblib.load(
    MODEL_PATH
)


# Use raw, unprocessed test rows
sample_data = (
    X_test_model
    .iloc[:5]
    .copy()
)


sample_predictions = (
    loaded_pipeline
    .predict(sample_data)
)


print(
    "\nReloaded Pipeline Test:"
)


print(
    "\nRaw Sample Data:"
)

print(
    sample_data
)


print(
    "\nPredictions:"
)

print(
    sample_predictions
)


print(
    "\nActual Values:"
)

print(
    y_test.iloc[:5].values
)


print(
    "\nThe saved artifact includes both preprocessing "
    "and the final estimator, and it can make predictions "
    "directly from raw new data."
)


print(
    "\n========== MODULE 2 PART B COMPLETED SUCCESSFULLY =========="
)