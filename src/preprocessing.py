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
    count_features = sorted(set(df.columns) - excluded)

    return {
        "nominal": [c for c in NOMINAL_FEATURES if c in df.columns],
        "binary": [c for c in BINARY_FEATURES if c in df.columns],
        "count": count_features,
        "ordinal": count_features,
        "continuous": [c for c in CONTINUOUS_FEATURES if c in df.columns],
    }


def handle_missing_values(df: pd.DataFrame, strategy: str = "median"):
    """Handle missing values and return the cleaned dataframe plus summary info."""
    result = df.copy()
    missing_per_column = result.isna().sum()
    info = {
        "total_missing": int(missing_per_column.sum()),
        "missing_per_column": missing_per_column.to_dict(),
        "strategy": strategy,
    }

    if info["total_missing"] == 0:
        return result, info

    if strategy == "drop":
        result = result.dropna()
    elif strategy in {"median", "mean"}:
        for col in result.columns:
            if not result[col].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(result[col]):
                fill_value = result[col].median() if strategy == "median" else result[col].mean()
            else:
                fill_value = result[col].mode(dropna=True).iloc[0]
            result[col] = result[col].fillna(fill_value)
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")

    return result, info


def encode_features(
    df: pd.DataFrame,
    feature_types: Optional[dict[str, list[str]]] = None,
    method: str = "onehot",
) -> pd.DataFrame:
    """Encode nominal features using one-hot or label encoding."""
    feature_types = feature_types or get_feature_types(df)
    result = df.copy()
    nominal_cols = [c for c in feature_types["nominal"] if c in result.columns]

    if method == "onehot":
        return pd.get_dummies(result, columns=nominal_cols, dtype=int)

    if method == "label":
        for col in nominal_cols:
            encoder = LabelEncoder()
            result[col] = encoder.fit_transform(result[col].astype(str))
        return result

    raise ValueError(f"Unknown encoding method: {method}")


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

# def standardize_features(
#     X: pd.DataFrame,
#     fit: bool = True,
#     scaler: Optional[StandardScaler] = None,
# ):
#     """Standardize all numeric columns except binary (0/1)."""
#     df = X.copy()
#     # Identify numeric columns
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     # Exclude binary columns (all values in {0, 1})
#     scale_cols = [col for col in numeric_cols if not set(df[col].dropna().unique()).issubset({0, 1})]

#     if not scale_cols:
#         return df, scaler

#     if fit:
#         scaler = StandardScaler()
#         scaled_values = scaler.fit_transform(df[scale_cols])
#         mode = "fit"
#     else:
#         if scaler is None:
#             raise ValueError("scaler must be provided when fit=False")
#         scaled_values = scaler.transform(df[scale_cols])
#         mode = "transform"

#     df = df.astype({col: float for col in scale_cols})
#     df.loc[:, scale_cols] = scaled_values
#     print(f"Standardized {len(scale_cols)} numeric features (excluding binary) ({mode}).")
#     return df, scaler


def standardize_feature_splits(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    binary_columns: Optional[list[str]] = None,
):
    """Fit a scaler on train and transform train/val/test, preserving binary columns."""
    binary_columns = binary_columns or [
        col for col in X_train.columns if set(X_train[col].dropna().unique()).issubset({0, 1})
    ]
    binary_set = set(binary_columns)
    scaled_columns = [col for col in X_train.columns if col not in binary_set]

    scaler = StandardScaler()

    def transform_split(X: Optional[pd.DataFrame], fit: bool = False):
        if X is None:
            return None
        result = X.copy().astype({col: float for col in scaled_columns})
        if fit:
            values = scaler.fit_transform(result[scaled_columns])
        else:
            values = scaler.transform(result[scaled_columns])
        result.loc[:, scaled_columns] = values
        return result

    X_train_scaled = transform_split(X_train, fit=True)
    X_val_scaled = transform_split(X_val)
    X_test_scaled = transform_split(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, scaled_columns
