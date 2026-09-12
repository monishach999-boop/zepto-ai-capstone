import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# =========================================================
# MODULE 2 - PART A: EDA, CLEANING AND DATA STORY
# =========================================================

# Get analytics folder path
BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# TASK 1: LOAD AND PROFILE THE TITANIC DATASET
# ---------------------------------------------------------

print("\n========== TASK 1: DATA LOADING AND PROFILING ==========")

# Load dataset only once from seaborn
df = sns.load_dataset("titanic")

# Save offline fallback immediately after loading
df.to_csv(BASE_DIR / "titanic.csv", index=False)

print("\nTitanic dataset saved as analytics/titanic.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe(include="all"))

# Missing value percentages
missing_percent = (df.isnull().sum() / len(df)) * 100

print("\nMissing Value Percentages:")
for column, percent in missing_percent.items():
    if percent > 0:
        print(f"{column}: {percent:.2f}%")


# ---------------------------------------------------------
# TASK 2: MISSING VALUE HANDLING
# Threshold rule:
# < 5%    -> drop rows
# 5-30%   -> impute
# > 30%   -> drop column / justify
# ---------------------------------------------------------

print("\n========== TASK 2: MISSING VALUE HANDLING ==========")

# AGE: around 20% missing -> impute with median
age_missing = missing_percent["age"]

if 5 <= age_missing <= 30:
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median)

    print(
        f"age: {age_missing:.2f}% missing -> "
        f"filled using median ({age_median:.2f})"
    )


# EMBARKED: less than 5% missing -> drop affected rows
embarked_missing = missing_percent["embarked"]

if embarked_missing < 5:
    df = df.dropna(subset=["embarked"])

    print(
        f"embarked: {embarked_missing:.2f}% missing -> "
        "rows with missing embarked values dropped"
    )


# EMBARK_TOWN: less than 5% missing -> drop affected rows
embark_town_missing = missing_percent["embark_town"]

if embark_town_missing < 5:
    df = df.dropna(subset=["embark_town"])

    print(
        f"embark_town: {embark_town_missing:.2f}% missing -> "
        "rows with missing embark_town values dropped"
    )


# DECK: very high missing percentage -> drop column
deck_missing = missing_percent["deck"]

if deck_missing > 30:
    df = df.drop(columns=["deck"])

    print(
        f"deck: {deck_missing:.2f}% missing -> "
        "column dropped because too much data is missing"
    )


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:")
print(df.shape)


# ---------------------------------------------------------
# TASK 3: UNIVARIATE ANALYSIS
# Histogram + boxplot for AGE and FARE
# IQR outlier count
# Fare mean, median and mode
# ---------------------------------------------------------

print("\n========== TASK 3: UNIVARIATE ANALYSIS ==========")


def analyze_numeric_column(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = data[
        (data[column] < lower_bound)
        | (data[column] > upper_bound)
    ]

    print(f"\n{column.upper()} Analysis")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Number of Outliers: {len(outliers)}")

    return len(outliers)


age_outliers = analyze_numeric_column(df, "age")
fare_outliers = analyze_numeric_column(df, "fare")


print("\nFare Statistics:")
fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print(f"Mean: {fare_mean:.2f}")
print(f"Median: {fare_median:.2f}")
print(f"Mode: {fare_mode:.2f}")


print("\nFare Distribution Interpretation:")

if fare_mean > fare_median > fare_mode:
    print(
        "Fare is strongly right-skewed because "
        "mean > median > mode."
    )
elif fare_mean < fare_median:
    print(
        "Fare shows left-skewness because "
        "the mean is below the median."
    )
else:
    print(
        "Fare is approximately symmetric because "
        "its mean and median are relatively close."
    )


# AGE histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# AGE boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="age")
plt.title("Age Boxplot")
plt.tight_layout()
plt.show()


# FARE histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="fare", kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# FARE boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="fare")
plt.title("Fare Boxplot")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# TASK 4: BIVARIATE ANALYSIS
# Boolean masking:
#   sex
#   pclass
#   sex + pclass
#
# Correlation matrix must contain exactly:
# survived, pclass, age, sibsp, parch, fare
# ---------------------------------------------------------

print("\n========== TASK 4: BIVARIATE ANALYSIS ==========")


# ---- Survival rate by sex using Boolean masks ----

male_mask = df["sex"] == "male"
female_mask = df["sex"] == "female"

male_survival = df.loc[male_mask, "survived"].mean()
female_survival = df.loc[female_mask, "survived"].mean()

print("\nSurvival Rate by Sex:")
print(f"Male: {male_survival:.2%}")
print(f"Female: {female_survival:.2%}")


# ---- Survival rate by passenger class ----

print("\nSurvival Rate by Passenger Class:")

for passenger_class in [1, 2, 3]:
    class_mask = df["pclass"] == passenger_class

    survival_rate = df.loc[
        class_mask,
        "survived"
    ].mean()

    print(
        f"Class {passenger_class}: "
        f"{survival_rate:.2%}"
    )


# ---- Survival rate by sex AND passenger class ----

print("\nSurvival Rate by Sex and Passenger Class:")

for gender in ["female", "male"]:
    for passenger_class in [1, 2, 3]:

        combined_mask = (
            (df["sex"] == gender)
            & (df["pclass"] == passenger_class)
        )

        survival_rate = df.loc[
            combined_mask,
            "survived"
        ].mean()

        print(
            f"{gender.capitalize()} - "
            f"Class {passenger_class}: "
            f"{survival_rate:.2%}"
        )


# ---- Exact six-column correlation matrix ----

numeric_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[numeric_columns].corr()

print("\n6 x 6 Correlation Matrix:")
print(correlation_matrix)


# Correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Heatmap")
plt.tight_layout()
plt.show()


# Find two strongest off-diagonal correlations
correlation_pairs = []

for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns)):

        col1 = numeric_columns[i]
        col2 = numeric_columns[j]

        value = correlation_matrix.loc[col1, col2]

        correlation_pairs.append(
            (col1, col2, value)
        )


correlation_pairs.sort(
    key=lambda x: abs(x[2]),
    reverse=True
)

strongest_two = correlation_pairs[:2]

print("\nTwo Strongest Off-Diagonal Correlations:")

for col1, col2, value in strongest_two:
    print(
        f"{col1} and {col2}: "
        f"{value:.3f}"
    )


print("\nCorrelation Interpretation:")

for col1, col2, value in strongest_two:

    if value > 0:
        direction = "positive"
    else:
        direction = "negative"

    print(
        f"{col1} and {col2} have a "
        f"{direction} relationship "
        f"with correlation {value:.3f}."
    )


# ---------------------------------------------------------
# TASK 5: MULTIVARIATE DATA STORY
# At least four distinct charts with interpretations
# ---------------------------------------------------------

print("\n========== TASK 5: MULTIVARIATE DATA STORY ==========")


# CHART 1 - Survival by Gender
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="sex",
    hue="survived"
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.show()

print("\nChart 1 Interpretation:")
print(
    "Female passengers had a considerably higher survival "
    "rate than male passengers."
)
print(
    "This suggests that gender was strongly associated "
    "with survival on the Titanic."
)


# CHART 2 - Survival by Passenger Class
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="pclass",
    hue="survived"
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.show()

print("\nChart 2 Interpretation:")
print(
    "First-class passengers had better survival outcomes "
    "than third-class passengers."
)
print(
    "Third-class passengers contained the largest number "
    "of non-survivors."
)


# CHART 3 - Age and Survival
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    hue="survived",
    kde=True,
    multiple="layer"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.show()

print("\nChart 3 Interpretation:")
print(
    "Most Titanic passengers were young and middle-aged adults."
)
print(
    "Survivors and non-survivors overlap across many ages, "
    "showing that age alone did not determine survival."
)


# CHART 4 - Fare by Passenger Class and Survival
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title("Fare by Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

print("\nChart 4 Interpretation:")
print(
    "First-class passengers generally paid much higher fares "
    "than second- and third-class passengers."
)
print(
    "Higher fares are associated with higher passenger class, "
    "which also showed better survival outcomes."
)


# ---------------------------------------------------------
# TASK 6: STANDARDIZATION SANITY CHECK
# This is ONLY for EDA.
# Modeling pipeline will perform its own train-only scaling.
# ---------------------------------------------------------

print("\n========== TASK 6: STANDARDIZATION CHECK ==========")

eda_scaler = StandardScaler()

scaled_values = eda_scaler.fit_transform(
    df[["age", "fare"]]
)

scaled_df = pd.DataFrame(
    scaled_values,
    columns=["age_scaled", "fare_scaled"],
    index=df.index
)


print("\nBefore Standardization:")
print(
    df[["age", "fare"]].agg(
        ["mean", "std"]
    )
)


print("\nAfter Standardization:")
print(
    scaled_df[
        ["age_scaled", "fare_scaled"]
    ].agg(["mean", "std"])
)


# AGE before and after
plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    kde=True,
    label="Age Before",
    stat="density"
)

sns.histplot(
    scaled_df["age_scaled"],
    kde=True,
    label="Age After",
    stat="density"
)

plt.title("Age: Before vs After Standardization")
plt.legend()
plt.tight_layout()
plt.show()


# FARE before and after
plt.figure(figsize=(8, 5))

sns.histplot(
    df["fare"],
    kde=True,
    label="Fare Before",
    stat="density"
)

sns.histplot(
    scaled_df["fare_scaled"],
    kde=True,
    label="Fare After",
    stat="density"
)

plt.title("Fare: Before vs After Standardization")
plt.legend()
plt.tight_layout()
plt.show()


print("\nStandardization Interpretation:")
print(
    "After standardization, age and fare have means "
    "approximately equal to 0."
)
print(
    "Their standard deviations are approximately equal to 1."
)
print(
    "This scaling is only an EDA sanity check. "
    "The modeling pipeline performs scaling separately "
    "using training data only."
)


print("\n========== PART A COMPLETED SUCCESSFULLY ==========")