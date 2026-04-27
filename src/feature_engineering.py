"""
Feature engineering utilities for Student Dropout and Academic Success dataset.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# Academic Features

def add_academic_features(df: pd.DataFrame):
    """Add academic performance derived features."""
    df = df.copy()

    e1 = df["Curricular units 1st sem (enrolled)"].replace(0, np.nan)
    e2 = df["Curricular units 2nd sem (enrolled)"].replace(0, np.nan)
    a1 = df["Curricular units 1st sem (approved)"]
    a2 = df["Curricular units 2nd sem (approved)"]
    ev1 = df["Curricular units 1st sem (evaluations)"]
    ev2 = df["Curricular units 2nd sem (evaluations)"]
    g1 = df["Curricular units 1st sem (grade)"]
    g2 = df["Curricular units 2nd sem (grade)"]
    c1 = df["Curricular units 1st sem (credited)"]
    c2 = df["Curricular units 2nd sem (credited)"]

    total_enrolled = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]

    df["approval_rate_1st"] = (a1 / e1).fillna(0).replace([np.inf, -np.inf], 0)
    df["approval_rate_2nd"] = (a2 / e2).fillna(0).replace([np.inf, -np.inf], 0)
    df["total_approved"] = a1 + a2
    df["total_enrolled"] = total_enrolled
    df["overall_approval_rate"] = (df["total_approved"] / total_enrolled.replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    df["grade_improvement"] = g2 - g1
    df["evaluation_intensity_1st"] = (ev1 / e1).fillna(0).replace([np.inf, -np.inf], 0)
    df["evaluation_intensity_2nd"] = (ev2 / e2).fillna(0).replace([np.inf, -np.inf], 0)
    df["grade_per_evaluation_1st"] = (g1 / ev1.replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    df["grade_per_evaluation_2nd"] = (g2 / ev2.replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    df["credited_ratio_1st"] = (c1 / e1).fillna(0).replace([np.inf, -np.inf], 0)
    df["credited_ratio_2nd"] = (c2 / e2).fillna(0).replace([np.inf, -np.inf], 0)

    return df


# Socioeconomic Features

def add_socioeconomic_features(df: pd.DataFrame):
    """Add socioeconomic derived features."""
    df = df.copy()

    mq = df["Mother's qualification"]
    fq = df["Father's qualification"]
    mo = df["Mother's occupation"]
    fo = df["Father's occupation"]

    df["parent_max_qualification"] = np.maximum(mq, fq)
    df["parent_avg_qualification"] = (mq + fq) / 2
    df["parent_qualification_gap"] = np.abs(mq - fq)
    df["parent_same_occupation"] = (mo == fo).astype(int)

    return df


# Financial Risk Features

def add_financial_risk_features(df: pd.DataFrame):
    """Add financial risk derived features."""
    df = df.copy()

    df["financial_risk"] = ((df["Debtor"] == 1) & (df["Tuition fees up to date"] == 0)).astype(int)
    df["has_scholarship_and_debt"] = ((df["Scholarship holder"] == 1) & (df["Debtor"] == 1)).astype(int)
    df["no_financial_stress"] = (
        (df["Scholarship holder"] == 1) & (df["Tuition fees up to date"] == 1) & (df["Debtor"] == 0)
    ).astype(int)

    return df


# Demographic Features

def add_demographic_features(df: pd.DataFrame):
    """Add age and demographic derived features."""
    df = df.copy()

    age = df["Age at enrollment"]
    df["is_mature_student"] = (age > 23).astype(int)
    df["is_traditional_student"] = (age <= 20).astype(int)
    df["is_very_mature"] = (age > 25).astype(int)

    return df


# Macroeconomic Features

def add_macroeconomic_features(df: pd.DataFrame):
    """Add macroeconomic derived features."""
    df = df.copy()

    df["economic_condition"] = df["GDP"] - df["Unemployment rate"] - df["Inflation rate"]

    return df


# Application Features

def add_application_features(df: pd.DataFrame):
    """Add application behavior derived features."""
    df = df.copy()

    order = df["Application order"]
    df["is_first_application"] = (order == 1).astype(int)
    df["is_late_application"] = (order > 3).astype(int)

    return df


# All Engineered Features

def add_all_engineered_features(df: pd.DataFrame):
    """Apply all feature engineering transformations."""
    df = add_academic_features(df)
    df = add_socioeconomic_features(df)
    df = add_financial_risk_features(df)
    df = add_demographic_features(df)
    df = add_macroeconomic_features(df)
    df = add_application_features(df)
    return df



# Polynomial Feature Interactions

def add_polynomial_interactions(
    X: pd.DataFrame,
    feature_names: list[str],
    degree: int = 2,
) -> pd.DataFrame:
    """Generate pairwise interaction terms for selected features.

    Only degree=2 pairwise products are created (not full polynomial expansion)
    to keep dimensionality manageable.
    """
    X = X.copy()
    valid_features = [f for f in feature_names if f in X.columns]

    new_cols = {}
    for i in range(len(valid_features)):
        for j in range(i + 1, len(valid_features)):
            f1, f2 = valid_features[i], valid_features[j]
            col_name = f"{f1}_x_{f2}"
            new_cols[col_name] = X[f1].values * X[f2].values

    if new_cols:
        interaction_df = pd.DataFrame(new_cols, index=X.index)
        X = pd.concat([X, interaction_df], axis=1)
        print(f"Added {len(new_cols)} interaction features from {len(valid_features)} base features")

    return X


# Target Encoding

def target_encode_column(train_series: pd.Series, test_series: pd.Series, y_train: pd.Series, smoothing: float = 10.0):
    """Target-encode a categorical column: replace each category with smoothed target mean."""
    global_mean = y_train.mean()

    temp = pd.DataFrame({"cat": train_series, "y": y_train})
    agg = temp.groupby("cat")["y"].agg(["mean", "count"])

    smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    encoding_map = smoothed.to_dict()

    train_enc = train_series.map(encoding_map).fillna(global_mean)
    test_enc = test_series.map(encoding_map).fillna(global_mean)

    return train_enc, test_enc


# Out-of-Fold Target Encoding (avoids training-set leakage)

def _target_encode_column_oof(train_series: pd.Series, y_train: pd.Series, n_folds: int, smoothing: float):
    """OOF target-encode a single column: each sample is encoded using K-1 folds."""
    from sklearn.model_selection import KFold

    train_enc = pd.Series(np.nan, index=train_series.index)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train_series):
        y_kf = y_train.iloc[train_idx]
        series_kf = train_series.iloc[train_idx]
        val_series = train_series.iloc[val_idx]

        global_mean = y_kf.mean()
        temp = pd.DataFrame({"cat": series_kf, "y": y_kf})
        agg = temp.groupby("cat")["y"].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        encoding_map = smoothed.to_dict()

        train_enc.iloc[val_idx] = val_series.map(encoding_map).fillna(global_mean)

    train_enc = train_enc.fillna(y_train.mean())
    return train_enc


def apply_target_encoding_oof(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, columns: list[str], n_folds: int = 5, smoothing: float = 10.0):
    """OOF target-encode multiple columns: fit K-fold on train, full mean on test."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in columns:
        if col not in X_train.columns:
            continue
        train_enc = _target_encode_column_oof(X_train[col], y_train, n_folds=n_folds, smoothing=smoothing)
        _, test_enc = target_encode_column(X_train[col], X_test[col], y_train, smoothing=smoothing)
        X_train[col] = train_enc.values
        X_test[col] = test_enc.values

    return X_train, X_test


# Feature Selection

def select_features_by_importance(X: pd.DataFrame, model, top_k: int = 60):
    """Select top-k features by importance from a fitted tree model."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(top_k).index.tolist()

    print(f"Selected top {top_k} features out of {X.shape[1]}")
    return X[top_features], top_features
