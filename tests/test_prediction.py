import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.prediction import (
    get_model,
    train_model,
    predict_and_evaluate,
    plot_confusion_matrix,
    compare_models,
    MODEL_REGISTRY,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


class TestGetModel:
    def test_returns_dt(self):
        m = get_model("dt")
        assert hasattr(m, "fit")

    def test_returns_lr(self):
        m = get_model("lr")
        assert m.max_iter == 10000  # TDD: will fail first (currently 5000), then fix

    def test_returns_svm(self):
        m = get_model("svm")
        assert m.kernel == "rbf"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model("bad")

    def test_override_params(self):
        m = get_model("dt", max_depth=3)
        assert m.max_depth == 3


class TestTrainModel:
    def test_trains_and_predicts(self, sample_features_np, sample_targets_np):
        model = train_model(sample_features_np, sample_targets_np, "dt")
        preds = model.predict(sample_features_np)
        assert len(preds) == 100


class TestPredictAndEvaluate:
    def test_returns_expected_keys(self, sample_features_np, sample_targets_np):
        model = train_model(sample_features_np, sample_targets_np, "dt")
        result = predict_and_evaluate(model, sample_features_np, sample_targets_np)
        assert "accuracy" in result
        assert "confusion_matrix" in result
        assert "report" in result
        assert 0 <= result["accuracy"] <= 1


class TestPlotConfusionMatrix:
    def test_returns_axes(self, sample_targets_np):
        preds = sample_targets_np.copy()
        ax = plot_confusion_matrix(sample_targets_np, preds, title="Test CM")
        assert isinstance(ax, plt.Axes)


class TestCompareModels:
    def test_returns_dataframe(self, sample_features_np, sample_targets_np):
        model = train_model(sample_features_np, sample_targets_np, "dt")
        result = predict_and_evaluate(model, sample_features_np, sample_targets_np, "test")
        df = compare_models({"DT": result})
        assert "Model" in df.columns
        assert "F1 (macro)" in df.columns
        assert len(df) == 1
