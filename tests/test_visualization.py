import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization import (
    compute_tsne,
    plot_tsne_2d,
    plot_tsne_3d,
    plot_perplexity_comparison,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


class TestComputeTsne:
    def test_2d_output_shape(self, sample_features_np):
        result = compute_tsne(sample_features_np, perplexity=5, n_components=2)
        assert result.shape == (100, 2)

    def test_3d_output_shape(self, sample_features_np):
        result = compute_tsne(sample_features_np, perplexity=5, n_components=3)
        assert result.shape == (100, 3)


class TestPlotTsne2d:
    def test_returns_axes(self, sample_features_np, sample_targets_np):
        embedding = compute_tsne(sample_features_np, perplexity=5)
        ax = plot_tsne_2d(embedding, sample_targets_np)
        assert isinstance(ax, plt.Axes)

    def test_save_to_file(self, sample_features_np, sample_targets_np, tmp_path):
        embedding = compute_tsne(sample_features_np, perplexity=5)
        path = str(tmp_path / "test_tsne.png")
        plot_tsne_2d(embedding, sample_targets_np, save_path=path)
        assert (tmp_path / "test_tsne.png").exists()


class TestPlotTsne3d:
    def test_returns_figure(self, sample_features_np, sample_targets_np):
        embedding = compute_tsne(sample_features_np, perplexity=5, n_components=3)
        fig = plot_tsne_3d(embedding, sample_targets_np)
        assert isinstance(fig, plt.Figure)


class TestPerplexityComparison:
    def test_returns_figure(self, sample_features_np, sample_targets_np):
        results = {}
        for p in [5, 10]:
            results[p] = compute_tsne(sample_features_np, perplexity=p)
        fig = plot_perplexity_comparison(results, sample_targets_np)
        assert isinstance(fig, plt.Figure)
