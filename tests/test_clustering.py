import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    apply_kmeans,
    apply_agglomerative,
    apply_dbscan,
    evaluate_clustering,
    find_optimal_kmeans,
    plot_elbow_method,
    plot_cluster_scatter,
    plot_algorithm_comparison,
    plot_cluster_size_distribution,
    compare_clusters_to_classes,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


class TestApplyKmeans:
    def test_returns_correct_labels(self, sample_features_np):
        labels, model = apply_kmeans(sample_features_np, n_clusters=3)
        assert len(labels) == 100
        assert set(labels) <= {0, 1, 2}

    def test_model_has_inertia(self, sample_features_np):
        _, model = apply_kmeans(sample_features_np, n_clusters=3)
        assert model.inertia_ > 0


class TestApplyAgglomerative:
    def test_returns_correct_labels(self, sample_features_np):
        labels, model = apply_agglomerative(sample_features_np, n_clusters=3)
        assert len(labels) == 100
        assert len(set(labels)) == 3


class TestApplyDbscan:
    def test_returns_labels(self, sample_features_np):
        labels, model = apply_dbscan(sample_features_np, eps=2.0, min_samples=3)
        assert len(labels) == 100


class TestEvaluateClustering:
    def test_unsupervised_metrics(self, sample_features_np):
        labels, _ = apply_kmeans(sample_features_np, n_clusters=3)
        metrics = evaluate_clustering(sample_features_np, labels)
        assert "silhouette" in metrics
        assert "calinski_harabasz" in metrics
        assert "davies_bouldin" in metrics

    def test_supervised_metrics(self, sample_features_np, sample_targets_np):
        labels, _ = apply_kmeans(sample_features_np, n_clusters=3)
        metrics = evaluate_clustering(sample_features_np, labels, y_true=sample_targets_np)
        assert "adjusted_rand_index" in metrics
        assert "normalized_mutual_info" in metrics

    def test_single_cluster_returns_empty(self, sample_features_np):
        labels = np.zeros(100, dtype=int)
        metrics = evaluate_clustering(sample_features_np, labels)
        assert metrics == {}


class TestFindOptimalKmeans:
    def test_returns_results_for_all_k(self, sample_features_np):
        results = find_optimal_kmeans(sample_features_np, k_range=range(2, 5))
        assert set(results.keys()) == {2, 3, 4}
        assert "inertia" in results[2]


class TestPlotDendrogram:
    def test_returns_figure(self, sample_features_np):
        from src.clustering import plot_dendrogram
        fig = plot_dendrogram(sample_features_np[:50], method="ward")
        assert isinstance(fig, plt.Figure)

    def test_save_to_file(self, sample_features_np, tmp_path):
        from src.clustering import plot_dendrogram
        path = str(tmp_path / "dendro.png")
        plot_dendrogram(sample_features_np[:50], method="ward", save_path=path)
        assert (tmp_path / "dendro.png").exists()


class TestRankAlgorithms:
    def test_normalized_ranking(self):
        from src.clustering import rank_algorithms
        all_results = {
            "A": {"silhouette": 0.5, "calinski_harabasz": 800, "davies_bouldin": 1.0,
                   "adjusted_rand_index": 0.3, "normalized_mutual_info": 0.3},
            "B": {"silhouette": 0.3, "calinski_harabasz": 600, "davies_bouldin": 2.0,
                   "adjusted_rand_index": 0.1, "normalized_mutual_info": 0.1},
        }
        ranked = rank_algorithms(all_results)
        assert isinstance(ranked, list)
        assert ranked[0][0] == "A"


class TestPlotFunctions:
    def test_elbow_method(self, sample_features_np):
        results = find_optimal_kmeans(sample_features_np, k_range=range(2, 5))
        fig = plot_elbow_method(results)
        assert isinstance(fig, plt.Figure)

    def test_cluster_scatter(self, sample_features_np, sample_targets_np):
        labels, _ = apply_kmeans(sample_features_np[:, :2], n_clusters=3)
        ax = plot_cluster_scatter(sample_features_np[:, :2], labels, "Test")
        assert isinstance(ax, plt.Axes)

    def test_cluster_size_distribution(self, sample_features_np):
        labels, _ = apply_kmeans(sample_features_np, n_clusters=3)
        fig = plot_cluster_size_distribution(labels, "Test")
        assert isinstance(fig, plt.Figure)

    def test_compare_clusters_to_classes(self, sample_targets_np):
        labels = np.random.RandomState(42).choice([0, 1, 2], 100)
        ct = compare_clusters_to_classes(labels, sample_targets_np)
        assert isinstance(ct, pd.DataFrame)
