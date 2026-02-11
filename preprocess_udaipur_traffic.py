import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("udaipur_traffic_dataset_2000_records.csv")

print("Initial Shape:", df.shape)

# -------------------------------
# 1. Remove duplicate records
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# 2. Handle missing values
# -------------------------------

# Numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# 3. Outlier treatment (IQR)
# -------------------------------
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# -------------------------------
# 4. Encode categorical features
# -------------------------------
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# -------------------------------
# 5. Standardize numeric features
# -------------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------
# 6. Save cleaned dataset
# -------------------------------
df.to_csv("udaipur_traffic_dataset_cleaned.csv", index=False)

print("Final Shape:", df.shape)
print("Preprocessing completed successfully!")
