# Module 2 - Part B: Predictive Modeling

import pandas as pd

# Load the cleaned dataset created in Part A
df = pd.read_csv("analytics/titanic.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
# Step 2: Train-Test Split

from sklearn.model_selection import train_test_split

# survived is our classification target
X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split:")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))
# Step 3: Preprocessing Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Features used for prediction
numeric_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Keep only the selected features
X_train_model = X_train[numeric_features + categorical_features]
X_test_model = X_test[numeric_features + categorical_features]

print("\nPreprocessing pipeline created successfully!")
print("Training features:", X_train_model.shape)
print("Testing features:", X_test_model.shape)
# Step 4: Train Classification Models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create models with preprocessing included
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train all three models
logistic_model.fit(X_train_model, y_train)
decision_tree_model.fit(X_train_model, y_train)
random_forest_model.fit(X_train_model, y_train)

print("\nThree classification models trained successfully!")
print("1. Logistic Regression")
print("2. Decision Tree")
print("3. Random Forest")
# Step 5: Evaluate Classification Models

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in models.items():

    y_pred = model.predict(X_test_model)
    y_prob = model.predict_proba(X_test_model)[:, 1]

    print("\n", "=" * 40)
    print(name)
    print("=" * 40)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))
   # Step 6: Visualize Decision Tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Get the trained Decision Tree from the pipeline
tree = decision_tree_model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = decision_tree_model.named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(20, 10))

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Titanic Survival")
plt.tight_layout()
plt.show()
# Step 7: Imbalance Handling Comparison

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("\n===== Imbalance Handling Comparison =====")

# Transform training and testing data using existing preprocessor
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)

# 1. Baseline Random Forest
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train_ready, y_train)
baseline_pred = baseline_model.predict(X_test_ready)

# 2. Random Forest with class_weight='balanced'
balanced_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

balanced_model.fit(X_train_ready, y_train)
balanced_pred = balanced_model.predict(X_test_ready)

# 3. Random Forest with SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_ready,
    y_train
)

smote_model = RandomForestClassifier(random_state=42)
smote_model.fit(X_train_smote, y_train_smote)
smote_pred = smote_model.predict(X_test_ready)

# Function to display scores
def show_scores(name, y_pred):
    print(f"\n{name}")
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred), 4))

show_scores("Baseline", baseline_pred)
show_scores("Class Weight Balanced", balanced_pred)
show_scores("SMOTE", smote_pred)
# Step 8: Hyperparameter Tuning using GridSearchCV

from sklearn.model_selection import GridSearchCV

print("\n===== Random Forest Hyperparameter Tuning =====")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

rf_tuning = RandomForestClassifier(
    random_state=42,
    oob_score=True
)

grid_search = GridSearchCV(
    estimator=rf_tuning,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_ready, y_train)

best_rf = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best CV F1 Score:", round(grid_search.best_score_, 4))
print("OOB Score:", round(best_rf.oob_score_, 4))
# Step 9: Regression - Predict Fare

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n===== Regression: Predicting Fare =====")

# Select age and fare
reg_data = df[["age", "fare"]].copy()

# Handle missing age values using median
reg_data["age"] = reg_data["age"].fillna(reg_data["age"].median())

X_reg = reg_data[["age"]]
y_reg = reg_data["fare"]

# Train-test split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)

# Predictions
y_reg_pred = reg_model.predict(X_reg_test)

# Regression metrics
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

n = len(y_reg_test)
p = X_reg_test.shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))
print("Adjusted R2:", round(adjusted_r2, 4))

# Residual Plot
residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_reg_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Prediction")
plt.tight_layout()
plt.show()
# Step 10: Final Model Comparison

import pandas as pd

classification_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [0.7989, 0.7709, 0.7877],
    "Precision": [0.7797, 0.7059, 0.7460],
    "Recall": [0.6667, 0.6957, 0.6812],
    "F1 Score": [0.7188, 0.7007, 0.7121],
    "ROC-AUC": [0.8194, 0.7514, 0.8205]
})

print("\n===== Classification Model Comparison =====")
print(classification_results)

regression_results = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MAE": [25.4461],
    "RMSE": [39.0466],
    "R2": [0.0147],
    "Adjusted R2": [0.0092]
})

print("\n===== Regression Model Results =====")
print(regression_results)
# Step 11: Final Recommendation and Save Pipeline

import joblib

print("\n===== Final Recommendation =====")
print("Logistic Regression is recommended as the final classification model.")
print("It provides the best overall balance of accuracy, F1 score, and ROC-AUC.")

# Create and fit the complete pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

final_pipeline.fit(X_train, y_train)

# Save complete pipeline
joblib.dump(final_pipeline, "analytics/titanic_survival_pipeline.joblib")

print("\nComplete pipeline saved successfully!")

# Reload the pipeline
loaded_pipeline = joblib.load("analytics/titanic_survival_pipeline.joblib")

# Test using raw, unprocessed data
sample_data = X_test.iloc[:5]

sample_predictions = loaded_pipeline.predict(sample_data)

print("\n===== Reloaded Pipeline Test =====")
print("Predictions:", sample_predictions)
print("Actual     :", y_test.iloc[:5].values)
print("\nPipeline reload and raw-data prediction successful!")