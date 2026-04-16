"""
Data visualization utilities for Student Dropout and Academic Success dataset.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

CLASS_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}


# ============================================================================
# t-SNE

def compute_tsne(X: np.ndarray, perplexity: int = 30, n_components: int = 2, random_state: int = 42, max_iter: int = 1000, learning_rate: str | float = "auto"):
    """Run t-SNE dimensionality reduction."""
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X)
    print(f"t-SNE done: perplexity={perplexity}, n_components={n_components}, shape={embedding.shape}")
    return embedding


# 2D Scatter Plots

def plot_tsne_2d(embedding: np.ndarray, labels: np.ndarray, perplexity: int = 30, save_path: Optional[str] = None, ax: Optional[plt.Axes] = None, show_legend: bool = True):
    """Plot 2D t-SNE scatter plot colored by class."""
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(10, 8))

    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=CLASS_COLORS.get(int(label), "#999999"),
            label=CLASS_NAMES.get(int(label), str(label)),
            alpha=0.6, s=10,
        )

    ax.set_title(f"t-SNE Projection (perplexity={perplexity})", fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    if show_legend:
        ax.legend(title="Class", markerscale=3, fontsize=10)

    if own_ax and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if own_ax:
        plt.tight_layout()

    return ax


# 3D Scatter Plot

def plot_tsne_3d(embedding: np.ndarray, labels: np.ndarray, perplexity: int = 30, save_path: Optional[str] = None):
    """Plot 3D t-SNE scatter plot colored by class."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
            c=CLASS_COLORS.get(int(label), "#999999"),
            label=CLASS_NAMES.get(int(label), str(label)),
            alpha=0.6, s=10,
        )

    ax.set_title(f"3D t-SNE Projection (perplexity={perplexity})", fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    ax.legend(title="Class", markerscale=3, fontsize=10)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.tight_layout()
    return fig


# Perplexity Comparison

def plot_perplexity_comparison(results: dict[int, np.ndarray], labels: np.ndarray, save_path: Optional[str] = None):
    """Plot grid comparing t-SNE results at different perplexity values."""
    perplexities = sorted(results.keys())
    n = len(perplexities)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, perp in enumerate(perplexities):
        row, col = idx // cols, idx % cols
        plot_tsne_2d(
            results[perp], labels, perplexity=perp,
            ax=axes[row, col], show_legend=(idx == 0),
        )

    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    plt.suptitle("t-SNE: Perplexity Comparison", fontweight="bold", y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.tight_layout()
    return fig
