"""
Data clustering utilities for Student Dropout and Academic Success dataset.

Provides functions for:
- K-Means, Agglomerative, and GMM clustering
- Cluster evaluation metrics
- Cluster visualization
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
)

# Color schemes
CLASS_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}  # Dropout, Enrolled, Graduate
CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

CLUSTER_COLORS = [
    "#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6",
    "#1abc9c", "#e91e63", "#00bcd4", "#ff5722", "#795548",
]


# Clustering Algorithms

def apply_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42):
    """Apply K-Means clustering and return labels and model."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, algorithm="lloyd")
    labels = model.fit_predict(X)
    return labels, model


def apply_agglomerative(X: np.ndarray, n_clusters: int, linkage: str = "ward"):
    """Apply Agglomerative Hierarchical Clustering and return labels and model."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model


def apply_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    """Apply DBSCAN clustering and return labels and model."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model


def apply_gmm(X: np.ndarray, n_components: int, covariance_type: str = "full", random_state: int = 42, n_init: int = 5):
    """Apply Gaussian Mixture Model clustering and return labels, model, and probabilities."""
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, n_init=n_init)
    model.fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)
    return labels, model, probs


# Cluster Evaluation Metrics

def evaluate_clustering(X: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray] = None):
    """Evaluate clustering quality with multiple metrics
    including silhouette, calinski_harabasz, davies_bouldin, adjusted_rand_index, normalized_mutual_info.
    """
    unique_labels = set(labels)
    non_noise_labels = unique_labels - {-1}
    if len(non_noise_labels) < 2:
        return {}

    metrics = {
        "silhouette": silhouette_score(X, labels),
        "silhouette_std": silhouette_samples(X, labels).std(),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "noise_ratio": 0.0,
    }

    if y_true is not None:
        metrics["adjusted_rand_index"] = adjusted_rand_score(y_true, labels)
        metrics["normalized_mutual_info"] = normalized_mutual_info_score(y_true, labels)

    return metrics


def find_optimal_kmeans(X: np.ndarray, k_range=range(2, 11), random_state: int = 42):
    """Evaluate K-Means over a range of k values."""
    results = {}
    for k in k_range:
        labels, model = apply_kmeans(X, n_clusters=k, random_state=random_state)
        metrics = evaluate_clustering(X, labels)
        results[k] = {"inertia": model.inertia_, **metrics}
    return results


def rank_algorithms(all_results: dict[str, dict[str, float]]):
    """Rank clustering algorithms by normalized metric scores."""
    if not all_results:
        return []

    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin", "adjusted_rand_index", "normalized_mutual_info"]
    scores = {name: 0.0 for name in all_results}
    for metric in metrics:
        values = np.array([result.get(metric, np.nan) for result in all_results.values()], dtype=float)
        if np.isnan(values).all() or np.nanmax(values) == np.nanmin(values):
            continue
        if metric == "davies_bouldin":
            normalized = (np.nanmax(values) - values) / (np.nanmax(values) - np.nanmin(values))
        else:
            normalized = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
        for name, score in zip(all_results, normalized):
            if not np.isnan(score):
                scores[name] += float(score)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# Visualization


def plot_cluster_scatter(X_2d: np.ndarray, labels: np.ndarray, title: str, y_true: Optional[np.ndarray] = None, save_path: Optional[str] = None, ax: Optional[plt.Axes] = None, show_legend: bool = True):
    """Plot 2D scatter plot of clusters."""
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        marker, name, alpha = "o", f"Cluster {label}", 0.6
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name, alpha=alpha, s=15, marker=marker)

    # Overlay true class boundaries
    if y_true is not None:
        for true_label in sorted(set(y_true)):
            mask = y_true == true_label
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                facecolors="none", edgecolors=CLASS_COLORS.get(int(true_label), "gray"),
                linewidth=1.0, s=30, alpha=0.3,
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if show_legend:
        ax.legend(title="Cluster", markerscale=2, fontsize=9)

    if own_ax and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if own_ax:
        plt.tight_layout()

    return ax


def plot_algorithm_comparison(all_results: dict[str, dict[str, float]], y_true: Optional[np.ndarray] = None, save_path: Optional[str] = None):
    """Plot bar charts comparing clustering algorithms across metrics."""
    # Define metrics to plot
    unsupervised = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    supervised = ["adjusted_rand_index", "normalized_mutual_info"]

    if y_true is not None:
        plot_metrics = unsupervised + supervised
        titles = {
            "silhouette": "Silhouette (higher)", "calinski_harabasz": "CH Index (higher)",
            "davies_bouldin": "DB Index (lower)", "adjusted_rand_index": "ARI (higher)",
            "normalized_mutual_info": "NMI (higher)",
        }
    else:
        plot_metrics = unsupervised
        titles = {
            "silhouette": "Silhouette (higher)", "calinski_harabasz": "CH Index (higher)",
            "davies_bouldin": "DB Index (lower)",
        }

    n = len(plot_metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    algorithms = list(all_results.keys())
    x_pos = np.arange(len(algorithms))
    colors = [CLUSTER_COLORS[i] for i in range(len(algorithms))]

    for ax, metric in zip(axes, plot_metrics):
        values = [all_results[alg].get(metric, np.nan) for alg in algorithms]
        bars = ax.bar(x_pos, values, color=colors, edgecolor="white")

        # Highlight best bar
        best_idx = np.nanargmin(values) if metric == "davies_bouldin" else np.nanargmax(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, rotation=30, ha="right")
        ax.set_title(titles.get(metric, metric), fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Clustering Algorithm Comparison", fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def plot_elbow_method(results: dict[int, dict[str, float]], save_path: Optional[str] = None):
    """Plot K-Means inertia across k values."""
    ks = sorted(results)
    inertias = [results[k]["inertia"] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, "o-", color="#3498db")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Method", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cluster_size_distribution(labels: np.ndarray, title: str, save_path: Optional[str] = None):
    """Plot cluster size distribution."""
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, color="#3498db", edgecolor="white")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Samples")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def compare_clusters_to_classes(labels: np.ndarray, y_true: np.ndarray):
    """Create cross-tabulation of clusters vs true class labels."""
    return pd.crosstab(pd.Series(labels, name="Cluster"), pd.Series(y_true, name="True Class"))


def plot_dendrogram(X: np.ndarray, method: str = "ward", max_d: Optional[float] = None, save_path: Optional[str] = None):
    """Plot hierarchical clustering dendrogram."""
    Z = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax,
    )
    if max_d is not None:
        ax.axhline(y=max_d, color="r", linestyle="--", label=f"Cut at d={max_d}")
        ax.legend()
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage)", fontweight="bold")
    ax.set_xlabel("Sample Index (or Cluster Size)")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig
