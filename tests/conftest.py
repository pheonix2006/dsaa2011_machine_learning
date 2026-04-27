import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features():
    """100 samples, 10 numeric features — mimics preprocessed data."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        rng.randn(100, 10),
        columns=[f"feat_{i}" for i in range(10)],
    )


@pytest.fixture
def sample_targets():
    """100 class labels: 0, 1, 2 (imbalanced like real data)."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.choice([0, 0, 1, 2, 2], size=100), name="Target")


@pytest.fixture
def raw_features():
    """DataFrame with columns matching real dataset feature names (subset)."""
    rng = np.random.RandomState(42)
    n = 80
    data = {
        "Marital Status": rng.choice([1, 2, 3, 4], n),
        "Application mode": rng.choice([1, 2, 3], n),
        "Course": rng.choice([1, 2, 3, 4, 5], n),
        "Nacionality": rng.choice([1, 2], n),
        "Mother's qualification": rng.choice(range(1, 6), n),
        "Father's qualification": rng.choice(range(1, 6), n),
        "Mother's occupation": rng.choice(range(1, 6), n),
        "Father's occupation": rng.choice(range(1, 6), n),
        "Daytime/evening attendance": rng.choice([0, 1], n),
        "Displaced": rng.choice([0, 1], n),
        "Educational special needs": rng.choice([0, 1], n),
        "Debtor": rng.choice([0, 1], n),
        "Tuition fees up to date": rng.choice([0, 1], n),
        "Gender": rng.choice([0, 1], n),
        "Scholarship holder": rng.choice([0, 1], n),
        "International": rng.choice([0, 1], n),
        "Previous qualification (grade)": rng.uniform(100, 180, n),
        "Admission grade": rng.uniform(100, 180, n),
        "Curricular units 1st sem (grade)": rng.uniform(0, 18, n),
        "Curricular units 2nd sem (grade)": rng.uniform(0, 18, n),
        "Unemployment rate": rng.uniform(7, 16, n),
        "Inflation rate": rng.uniform(-1, 3, n),
        "GDP": rng.uniform(-4, 3, n),
        "Age at enrollment": rng.randint(17, 50, n),
        "Application order": rng.randint(1, 6, n),
        "Curricular units 1st sem (enrolled)": rng.randint(0, 10, n),
        "Curricular units 2nd sem (enrolled)": rng.randint(0, 10, n),
        "Curricular units 1st sem (approved)": rng.randint(0, 8, n),
        "Curricular units 2nd sem (approved)": rng.randint(0, 8, n),
        "Curricular units 1st sem (evaluations)": rng.randint(0, 15, n),
        "Curricular units 2nd sem (evaluations)": rng.randint(0, 15, n),
        "Curricular units 1st sem (credited)": rng.randint(0, 5, n),
        "Curricular units 2nd sem (credited)": rng.randint(0, 5, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def raw_targets():
    """80 labels matching raw_features size."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.choice([0, 1, 2], size=80), name="Target")


@pytest.fixture
def sample_features_np(sample_features):
    """Numpy array version of sample_features."""
    return sample_features.values


@pytest.fixture
def sample_targets_np(sample_targets):
    """Numpy array version of sample_targets."""
    return sample_targets.values
