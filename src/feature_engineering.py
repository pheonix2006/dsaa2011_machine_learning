"""
Feature engineering utilities for Student Dropout and Academic Success dataset.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# ============================================================================
# Academic Features
# ============================================================================


def add_academic_features(df: pd.DataFrame) -> pd.DataFrame:
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


# ============================================================================
# Socioeconomic Features
# ============================================================================


def add_socioeconomic_features(df: pd.DataFrame) -> pd.DataFrame:
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


# ============================================================================
# Financial Risk Features
# ============================================================================


def add_financial_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add financial risk derived features."""
    df = df.copy()

    df["financial_risk"] = ((df["Debtor"] == 1) & (df["Tuition fees up to date"] == 0)).astype(int)
    df["has_scholarship_and_debt"] = ((df["Scholarship holder"] == 1) & (df["Debtor"] == 1)).astype(int)
    df["no_financial_stress"] = (
        (df["Scholarship holder"] == 1) & (df["Tuition fees up to date"] == 1) & (df["Debtor"] == 0)
    ).astype(int)

    return df


# ============================================================================
# Demographic Features
# ============================================================================


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age and demographic derived features."""
    df = df.copy()

    age = df["Age at enrollment"]
    df["is_mature_student"] = (age > 23).astype(int)
    df["is_traditional_student"] = (age <= 20).astype(int)
    df["is_very_mature"] = (age > 25).astype(int)

    return df


# ============================================================================
# Macroeconomic Features
# ============================================================================


def add_macroeconomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add macroeconomic derived features."""
    df = df.copy()

    df["economic_condition"] = df["GDP"] - df["Unemployment rate"] - df["Inflation rate"]

    return df


# ============================================================================
# Application Features
# ============================================================================


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add application behavior derived features."""
    df = df.copy()

    order = df["Application order"]
    df["is_first_application"] = (order == 1).astype(int)
    df["is_late_application"] = (order > 3).astype(int)

    return df


# ============================================================================
# All Engineered Features
# ============================================================================


def add_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = add_academic_features(df)
    df = add_socioeconomic_features(df)
    df = add_financial_risk_features(df)
    df = add_demographic_features(df)
    df = add_macroeconomic_features(df)
    df = add_application_features(df)
    return df


# ============================================================================
# Target Encoding
# ============================================================================


def target_encode_column(
    train_series: pd.Series,
    test_series: pd.Series,
    y_train: pd.Series,
    smoothing: float = 10.0,
) -> tuple[pd.Series, pd.Series]:
    """Target-encode a categorical column: replace each category with smoothed target mean."""
    global_mean = y_train.mean()

    temp = pd.DataFrame({"cat": train_series, "y": y_train})
    agg = temp.groupby("cat")["y"].agg(["mean", "count"])

    smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    encoding_map = smoothed.to_dict()

    train_enc = train_series.map(encoding_map).fillna(global_mean)
    test_enc = test_series.map(encoding_map).fillna(global_mean)

    return train_enc, test_enc


def apply_target_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target-encode multiple columns (fit on train, transform both)."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in columns:
        if col not in X_train.columns:
            continue
        train_enc, test_enc = target_encode_column(X_train[col], X_test[col], y_train, smoothing=smoothing)
        X_train[col] = train_enc.values
        X_test[col] = test_enc.values

    return X_train, X_test


# ============================================================================
# Binning
# ============================================================================


def bin_continuous(df: pd.DataFrame, column: str, bins: int | list[float], labels: list[str] | None = None) -> pd.DataFrame:
    """Bin a continuous column into discrete intervals."""
    df = df.copy()
    df[f"{column}_bin"] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    return df


# ============================================================================
# Feature Selection
# ============================================================================


def select_features_by_importance(X: pd.DataFrame, model, top_k: int = 60) -> tuple[pd.DataFrame, list]:
    """Select top-k features by importance from a fitted tree model."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(top_k).index.tolist()

    print(f"Selected top {top_k} features out of {X.shape[1]}")
    return X[top_features], top_features


def remove_low_variance_features(X: pd.DataFrame, threshold: float = 0.01) -> tuple[pd.DataFrame, list]:
    """Remove features with variance below threshold."""
    variances = X.var()
    low_var = variances[variances < threshold]
    removed = list(low_var.index)

    print(f"Removed {len(removed)} low-variance features (var < {threshold})")
    return X.drop(columns=removed), removed


def remove_highly_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list]:
    """Remove one of each pair with correlation above threshold."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"Removed {len(to_drop)} highly correlated features (|r| > {threshold})")
    return X.drop(columns=to_drop), to_drop


def select_top_k_features(X: pd.DataFrame, y: pd.Series, k: int = 30) -> tuple[pd.DataFrame, list]:
    """Select top k features by mutual information."""
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)

    selected = list(X.columns[selector.get_support()])

    print(f"Selected top {k} features by mutual information")
    return X[selected], selected
