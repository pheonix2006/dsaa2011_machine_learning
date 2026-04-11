"""
Data preprocessing utilities for Student Dropout and Academic Success dataset.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================================
# Feature type classification
# ============================================================================

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


def get_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify features into nominal, binary, count, continuous groups."""
    excluded = set(NOMINAL_FEATURES + BINARY_FEATURES + CONTINUOUS_FEATURES)
    count_features = sorted(set(df.columns) - excluded)

    return {
        "nominal": [c for c in NOMINAL_FEATURES if c in df.columns],
        "binary": [c for c in BINARY_FEATURES if c in df.columns],
        "count": count_features,
        "continuous": [c for c in CONTINUOUS_FEATURES if c in df.columns],
    }


# ============================================================================
# Data loading
# ============================================================================


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.Series]:
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


# ============================================================================
# Missing values
# ============================================================================


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> tuple[pd.DataFrame, dict]:
    """Fill or drop missing values in the DataFrame."""
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

    print(f"Found {missing.sum()} missing values in {len(missing_cols)} columns: {missing_cols.to_dict()}")

    df_clean = df.copy()

    if strategy == "drop":
        df_clean = df_clean.dropna()
        print(f"Dropped rows. New shape: {df_clean.shape}")
    else:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        fill_value = df_clean[numeric_cols].median() if strategy == "median" else df_clean[numeric_cols].mean()
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(fill_value)
        print(f"Filled missing values with {strategy}.")

    return df_clean, info


# ============================================================================
# Feature encoding
# ============================================================================


def encode_features(
    X: pd.DataFrame,
    feature_types: Optional[dict[str, list[str]]] = None,
    method: str = "onehot",
) -> pd.DataFrame:
    """Encode nominal categorical features (one-hot or label)."""
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


# ============================================================================
# Standardization
# ============================================================================


def standardize_features(
    X: pd.DataFrame,
    feature_types: Optional[dict[str, list[str]]] = None,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize continuous and count features to zero mean, unit variance."""
    if feature_types is None:
        feature_types = get_feature_types(X)

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


# ============================================================================
# Full pipeline
# ============================================================================


def preprocess_pipeline(
    data_dir: str = "data",
    encode_method: str = "onehot",
    do_standardize: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Run full preprocessing: load → missing → encode → standardize."""
    # 1. Load
    X, y = load_data(data_dir)

    # 2. Handle missing
    X, missing_info = handle_missing_values(X)

    # 3. Identify feature types
    feature_types = get_feature_types(X)
    ft = feature_types
    print(f"\nFeature types: nominal={len(ft['nominal'])}, binary={len(ft['binary'])}, "
          f"count={len(ft['count'])}, continuous={len(ft['continuous'])}")

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
