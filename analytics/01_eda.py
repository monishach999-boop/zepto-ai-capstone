import seaborn as sns
import pandas as pd

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

# Save a local copy inside analytics folder
df.to_csv("analytics/titanic.csv", index=False)

print("\nTitanic dataset saved successfully.")
# Summary statistics
print("\nSummary statistics:")
print(df.describe(include="all"))

# Missing value percentage
missing_percent = (df.isnull().sum() / len(df)) * 100

print("\nMissing value percentages:")

for column, percent in missing_percent.items():
    if percent > 0:
        print(column, ":", round(percent, 2), "%")

# Fill age with median
df["age"] = df["age"].fillna(df["age"].median())

# Fill embarked and embark_town with mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# Drop deck because it has too many missing values
df = df.drop(columns=["deck"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())
# Step 5: Univariate Analysis using IQR

for column in ["age", "fare"]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"\n{column.upper()} Analysis")
    print("Mean:", df[column].mean())
    print("Median:", df[column].median())
    print("Mode:", df[column].mode()[0])
    print("Number of outliers:", len(outliers))
    # Interpretation of distributions

print("\nInterpretation:")
print("Age is slightly right-skewed because the mean is greater than the median.")
print("Fare is strongly right-skewed because the mean is much greater than the median.")
# Step 6: Bivariate Analysis - Correlation Matrix

numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]

correlation = df[numeric_cols].corr()

print("\nCorrelation Matrix:")
print(correlation)

# Find strongest correlations excluding self-correlation
corr_pairs = correlation.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1].sort_values(ascending=False)

print("\nTwo strongest correlation pairs:")
print(corr_pairs.drop_duplicates().head(2))
# Step 7: Bivariate Analysis - Plots

import matplotlib.pyplot as plt
import seaborn as sns

# Survival by gender
sns.countplot(data=df, x="sex", hue="survived")
plt.title("Survival by Gender")
plt.show()

# Survival by passenger class
sns.countplot(data=df, x="pclass", hue="survived")
plt.title("Survival by Passenger Class")
plt.show()
# Step 8: More EDA Visualizations

# Age distribution
sns.histplot(data=df, x="age", hue="survived", kde=True)
plt.title("Age Distribution by Survival")
plt.show()

# Fare distribution by passenger class
sns.boxplot(data=df, x="pclass", y="fare")
plt.title("Fare Distribution by Passenger Class")
plt.show()
# Step 9: Standardization Check

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df["age_scaled"] = scaler.fit_transform(df[["age"]])
df["fare_scaled"] = scaler.fit_transform(df[["fare"]])

print("\nBefore Standardization:")
print(df[["age", "fare"]].describe())

print("\nAfter Standardization:")
print(df[["age_scaled", "fare_scaled"]].describe())
# Step 10: Before and After Standardization Plot

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

sns.histplot(df["age"], ax=axes[0, 0], kde=True)
axes[0, 0].set_title("Age - Before Standardization")

sns.histplot(df["age_scaled"], ax=axes[0, 1], kde=True)
axes[0, 1].set_title("Age - After Standardization")

sns.histplot(df["fare"], ax=axes[1, 0], kde=True)
axes[1, 0].set_title("Fare - Before Standardization")

sns.histplot(df["fare_scaled"], ax=axes[1, 1], kde=True)
axes[1, 1].set_title("Fare - After Standardization")

plt.tight_layout()
plt.show()
# Step 11: EDA Chart Interpretations

print("\nEDA Chart Interpretations:")

print("1. Survival by Gender:")
print("Female passengers had a much higher survival rate than male passengers.")

print("\n2. Survival by Passenger Class:")
print("First-class passengers had better survival, while third-class passengers had the highest number of deaths.")

print("\n3. Age Distribution by Survival:")
print("Most passengers were young adults, and both survivors and non-survivors were concentrated around this age group.")

print("\n4. Fare Distribution by Passenger Class:")
print("First-class passengers generally paid higher fares, while second and third-class passengers paid lower fares.")