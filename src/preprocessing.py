"""
Data preprocessing utilities
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


NOMINAL_FEATURES = [
    "Marital Status",
    "Application mode",
    "Course",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

BINARY_FEATURES = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

CONTINUOUS_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


def get_feature_types(df: pd.DataFrame):
    """Classify features into nominal, binary, count, continuous groups."""
    excluded = set(NOMINAL_FEATURES + BINARY_FEATURES + CONTINUOUS_FEATURES)
    ordinal_features = sorted(set(df.columns) - excluded)

    return {
        "nominal": [c for c in NOMINAL_FEATURES if c in df.columns],
        "binary": [c for c in BINARY_FEATURES if c in df.columns],
        "ordinal": ordinal_features,
        "continuous": [c for c in CONTINUOUS_FEATURES if c in df.columns],
    }


# Data loading

def load_data(data_dir: str = "data"):
    """Load features and targets from CSV files."""
    features_path = os.path.join(data_dir, "features.csv")
    targets_path = os.path.join(data_dir, "targets.csv")

    X = pd.read_csv(features_path)
    y_raw = pd.read_csv(targets_path).iloc[:, 0]

    le = LabelEncoder()
    y = pd.Series(np.array(le.fit_transform(y_raw), dtype=np.int64), name="Target", index=X.index)

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Target distribution:\n{y_raw.value_counts().to_string()}")

    return X, y

# Standardization

def standardize_features(X: pd.DataFrame, scaler: Optional[StandardScaler] = None):
    """Standardize continuous and count features to zero mean, unit variance."""
    df = X.copy()
    scaler = StandardScaler()
    arr = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(arr, columns=df.columns, index=df.index)
    print(f"Standardized {df_scaled.shape[1]} numerical features (fit).")
    return df_scaled, scaler
