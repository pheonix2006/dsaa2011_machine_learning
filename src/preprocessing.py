"""
Data preprocessing utilities for Student Dropout and Academic Success dataset.

This module provides functions for:
- Loading and inspecting the dataset
- Classifying feature types (numerical, binary, nominal)
- Handling missing values
- Encoding categorical features (one-hot / label)
- Standardizing numerical features
- Running the full preprocessing pipeline
"""

import os
from typing import Optional

from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Feature type classification
# ---------------------------------------------------------------------------

# Nominal categorical features: encoded as integers but represent categories
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

# Binary features: already encoded as 0/1
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

# Continuous numerical features (float values with high cardinality)
CONTINUOUS_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


def _get_count_features(df: pd.DataFrame) -> list[str]:
    """Identify count features (int columns not in BINARY or NOMINAL)."""
    all_cols = set(df.columns)
    excluded = set(NOMINAL_FEATURES + BINARY_FEATURES + CONTINUOUS_FEATURES)
    return sorted(all_cols - excluded)


def get_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify features into numerical, binary, and nominal groups.

    Returns a dict with keys: 'nominal', 'binary', 'count', 'continuous'.
    """
    return {
        "nominal": [c for c in NOMINAL_FEATURES if c in df.columns],
        "binary": [c for c in BINARY_FEATURES if c in df.columns],
        "count": _get_count_features(df),
        "continuous": [c for c in CONTINUOUS_FEATURES if c in df.columns],
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.Series]:
    """Load features and targets from CSV files.

    Returns:
        (X, y) where X is the feature DataFrame and y is the target Series
        with label-encoded target values.
    """
    features_path = os.path.join(data_dir, "features.csv")
    targets_path = os.path.join(data_dir, "targets.csv")

    X = pd.read_csv(features_path)
    y_raw = pd.read_csv(targets_path).iloc[:, 0]

    # Encode target labels to integers
    le = LabelEncoder()
    y = pd.Series(
        np.array(le.fit_transform(y_raw), dtype=np.int64),
        name="Target",
        index=X.index,
    )

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    class_map = dict(zip(
        np.asarray(le.classes_).tolist(),
        np.asarray(le.transform(le.classes_)).tolist(),
    ))
    print(f"Target classes: {class_map}")
    print(f"Target distribution:\n{y_raw.value_counts().to_string()}")

    return X, y


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
) -> tuple[pd.DataFrame, dict]:
    """Handle missing values in the DataFrame.

    Args:
        df: Input DataFrame.
        strategy: Fill strategy — 'median', 'mean', or 'drop'.

    Returns:
        (df_cleaned, info) where info contains missing value statistics.
    """
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    info = {
        "total_missing": int(missing.sum()),
        "missing_per_column": missing_cols.to_dict(),
        "strategy": strategy,
    }

    if missing.sum() == 0:
        print("No missing values found.")
        return df.copy(), info

    print(f"Found {missing.sum()} missing values in {len(missing_cols)} columns.")
    print(missing_cols)

    df_clean = df.copy()

    if strategy == "drop":
        df_clean = df_clean.dropna()
        print(f"Dropped rows with missing values. New shape: {df_clean.shape}")
    else:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        fill_value = df_clean[numeric_cols].median() if strategy == "median" else df_clean[numeric_cols].mean()
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(fill_value)
        print(f"Filled missing values with {strategy}.")

    return df_clean, info


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_features(
    X: pd.DataFrame,
    feature_types: Optional[dict[str, list[str]]] = None,
    method: str = "onehot",
) -> pd.DataFrame:
    """Encode nominal categorical features.

    Args:
        X: Feature DataFrame.
        feature_types: Output of get_feature_types(). If None, auto-detects.
        method: 'onehot' for one-hot encoding, 'label' for label encoding.

    Returns:
        Encoded DataFrame.
    """
    if feature_types is None:
        feature_types = get_feature_types(X)

    nominal_cols = feature_types["nominal"]
    df = X.copy()

    if not nominal_cols:
        print("No nominal features to encode.")
        return df

    if method == "onehot":
        df = pd.get_dummies(df, columns=nominal_cols, prefix=nominal_cols, dtype=int)
        print(f"One-hot encoded {len(nominal_cols)} nominal features. New shape: {df.shape}")
    elif method == "label":
        le = LabelEncoder()
        for col in nominal_cols:
            df[col] = np.asarray(le.fit_transform(df[col].astype(str)), dtype=np.int64)
        print(f"Label encoded {len(nominal_cols)} nominal features.")
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return df


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def standardize_features(
    X: pd.DataFrame,
    feature_types: Optional[dict[str, list[str]]] = None,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerical features (continuous + count) to zero mean, unit variance.

    Binary and one-hot encoded features are left unchanged.

    Args:
        X: Feature DataFrame.
        feature_types: Output of get_feature_types(). If None, auto-detects.
        fit: Whether to fit the scaler (True for training, False for testing).
        scaler: Pre-fitted scaler for transform-only mode.

    Returns:
        (X_scaled, scaler).
    """
    if feature_types is None:
        feature_types = get_feature_types(X)

    # Determine which columns to scale: continuous + count (not binary, not one-hot residuals)
    # After one-hot encoding, original nominal columns are gone, but new dummy columns exist.
    # We scale only known continuous columns and count columns that still exist.
    scale_cols = feature_types["continuous"] + feature_types["count"]
    scale_cols = [c for c in scale_cols if c in X.columns]

    if not scale_cols:
        print("No numerical features to standardize.")
        return X.copy(), scaler or StandardScaler()

    df = X.copy()

    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        print(f"Standardized {len(scale_cols)} numerical features (fit).")
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        df[scale_cols] = scaler.transform(df[scale_cols])
        print(f"Standardized {len(scale_cols)} numerical features (transform).")

    return df, scaler


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    data_dir: str = "data",
    encode_method: str = "onehot",
    do_standardize: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Run the full preprocessing pipeline.

    Steps: load → handle missing → encode → standardize.

    Returns:
        (X_processed, y, meta) where meta contains preprocessing metadata.
    """
    # 1. Load
    X, y = load_data(data_dir)

    # 2. Handle missing
    X, missing_info = handle_missing_values(X)

    # 3. Identify feature types (before encoding)
    feature_types = get_feature_types(X)
    print(f"\nFeature types: nominal={len(feature_types['nominal'])}, "
          f"binary={len(feature_types['binary'])}, "
          f"count={len(feature_types['count'])}, "
          f"continuous={len(feature_types['continuous'])}")

    # 4. Encode
    X = encode_features(X, feature_types, method=encode_method)

    # 5. Standardize
    scaler = None
    if do_standardize:
        X, scaler = standardize_features(X, feature_types, fit=True)

    meta = {
        "feature_types": feature_types,
        "missing_info": missing_info,
        "encode_method": encode_method,
        "standardized": do_standardize,
        "scaler": scaler,
        "original_columns": list(X.columns),
    }

    print(f"\nFinal shape: {X.shape}")
    return X, y, meta
