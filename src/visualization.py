"""
Data visualization utilities for Student Dropout and Academic Success dataset.

This module provides functions for:
- t-SNE dimensionality reduction and visualization
- 2D and 3D scatter plots with class labels
- Multi-parameter comparison grids
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

# Class color scheme (consistent across all notebooks)
# 类别配色方案（所有 Notebook 保持一致）
CLASS_COLORS = {
    0: "#e74c3c",  # Dropout - red
    1: "#f39c12",  # Enrolled - orange
    2: "#2ecc71",  # Graduate - green
}

CLASS_NAMES = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate",
}


def compute_tsne(
    X: np.ndarray,
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 42,
    max_iter: int = 1000,
    learning_rate: str | float = "auto",
) -> np.ndarray:
    """Run t-SNE dimensionality reduction.

    Args:
        X: Input feature matrix (n_samples, n_features).
        perplexity: t-SNE perplexity parameter. Typical range: 5-50.
        n_components: Target dimensionality (2 or 3).
        random_state: Random seed for reproducibility.
        max_iter: Maximum number of iterations.
        learning_rate: Learning rate. 'auto' uses max(200, n_samples / 12).

    Returns:
        Embedding coordinates of shape (n_samples, n_components).
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X)
    print(f"t-SNE done: perplexity={perplexity}, n_components={n_components}, "
          f"shape={embedding.shape}")
    return embedding


def plot_tsne_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot 2D t-SNE scatter plot colored by class.

    Args:
        embedding: t-SNE coordinates (n_samples, 2).
        labels: Class labels (n_samples,).
        perplexity: Perplexity value used (for title).
        save_path: If provided, save figure to this path.
        ax: Existing axes to plot on. If None, creates new figure.
        show_legend: Whether to display legend.

    Returns:
        The matplotlib Axes object.
    """
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(np.unique(labels))
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=CLASS_COLORS.get(int(label), "#999999"),
            label=CLASS_NAMES.get(int(label), str(label)),
            alpha=0.6,
            s=10,
            edgecolors="none",
        )

    ax.set_title(f"t-SNE Projection (perplexity={perplexity})", fontsize=14, fontweight="bold")
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


def plot_tsne_3d(
    embedding: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 3D t-SNE scatter plot colored by class.

    Args:
        embedding: t-SNE coordinates (n_samples, 3).
        labels: Class labels (n_samples,).
        perplexity: Perplexity value used (for title).
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = sorted(np.unique(labels))
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=CLASS_COLORS.get(int(label), "#999999"),
            label=CLASS_NAMES.get(int(label), str(label)),
            alpha=0.6,
            s=10,
            edgecolors="none",
        )

    ax.set_title(f"3D t-SNE Projection (perplexity={perplexity})", fontsize=14, fontweight="bold")
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


def plot_perplexity_comparison(
    results: dict[int, np.ndarray],
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 2x2 grid comparing t-SNE results at different perplexity values.

    Args:
        results: Dict mapping perplexity values to 2D embeddings.
        labels: Class labels (n_samples,).
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
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

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    plt.suptitle("t-SNE: Perplexity Comparison", fontsize=16, fontweight="bold", y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.tight_layout()
    return fig
