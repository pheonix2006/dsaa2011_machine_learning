import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import (
    calculate_metrics,
    plot_roc_curves,
    cross_validate_model,
    plot_validation_curve,
    plot_learning_curve,
    metrics_comparison_table,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


class TestCalculateMetrics:
    def test_returns_all_keys(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        metrics = calculate_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_weighted" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1])
        metrics = calculate_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0


class TestPlotRocCurves:
    def test_returns_figure(self, sample_features_np, sample_targets_np):
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(sample_features_np, sample_targets_np)
        fig = plot_roc_curves({"DT": model}, sample_features_np, sample_targets_np)
        assert isinstance(fig, plt.Figure)


class TestCrossValidateModel:
    def test_returns_expected_keys(self, sample_features_np, sample_targets_np):
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        result = cross_validate_model(model, sample_features_np, sample_targets_np, cv=3)
        assert "mean" in result
        assert "std" in result
        assert len(result["scores"]) == 3


class TestPlotValidationCurve:
    def test_returns_figure(self, sample_features_np, sample_targets_np):
        model = DecisionTreeClassifier(random_state=42)
        fig = plot_validation_curve(
            model, sample_features_np, sample_targets_np,
            param_name="max_depth", param_range=[2, 5, 10], cv=3,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotLearningCurve:
    def test_returns_figure(self, sample_features_np, sample_targets_np):
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        fig = plot_learning_curve(model, sample_features_np, sample_targets_np, cv=3)
        assert isinstance(fig, plt.Figure)


class TestMetricsComparisonTable:
    def test_returns_dataframe(self):
        results = {
            "DT": {"accuracy": 0.8, "f1_macro": 0.75},
            "LR": {"accuracy": 0.7, "f1_macro": 0.65},
        }
        df = metrics_comparison_table(results)
        assert len(df) == 2
        assert "Model" in df.columns


class TestTuneModel:
    def test_returns_best_estimator(self, sample_features_np, sample_targets_np):
        from src.evaluation import tune_model
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {"max_depth": [2, 5]}
        result = tune_model(model, param_grid, sample_features_np, sample_targets_np, cv=3)
        assert "best_estimator" in result
        assert "best_params" in result
        assert "best_score" in result
        assert hasattr(result["best_estimator"], "predict")

    def test_best_params_in_grid(self, sample_features_np, sample_targets_np):
        from src.evaluation import tune_model
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {"max_depth": [2, 5, 10]}
        result = tune_model(model, param_grid, sample_features_np, sample_targets_np, cv=3)
        assert result["best_params"]["max_depth"] in [2, 5, 10]
