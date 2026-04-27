"""
Prediction utilities for Student Dropout and Academic Success dataset.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
MODEL_REGISTRY = {
    "dt": lambda **kw: DecisionTreeClassifier(**{**{"random_state": 42, "max_depth": 10, "class_weight": "balanced"}, **kw}),
    "lr": lambda **kw: LogisticRegression(**{**{"random_state": 42, "max_iter": 50000, "solver": "saga", "class_weight": "balanced"}, **kw}),
    "svm": lambda **kw: SVC(**{**{"random_state": 42, "kernel": "rbf", "probability": True, "class_weight": "balanced"}, **kw}),
}


def get_model(model_type: str, **kwargs):
    """Return an unfitted sklearn classifier by type key."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str, **kwargs):
    """Train a classifier and return the fitted model."""
    model = get_model(model_type, **kwargs)
    model.fit(X_train, y_train)
    return model


def predict_and_evaluate(model, X: np.ndarray, y: np.ndarray, set_name: str = "test"):
    """Predict and return evaluation dict with accuracy, report, confusion matrix."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=list(CLASS_NAMES.values()), output_dict=True)
    cm = confusion_matrix(y, y_pred)
    print(f"[{set_name}] Accuracy: {acc:.4f}")
    return {"set_name": set_name, "accuracy": acc, "report": report, "confusion_matrix": cm, "y_pred": y_pred}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    labels: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Plot confusion matrix heatmap."""
    if labels is None:
        labels = list(CLASS_NAMES.values())
    cm = confusion_matrix(y_true, y_pred)
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontweight="bold")
    if own_ax:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
    return ax


def plot_decision_boundary(
    model,
    X_2d: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    save_path: Optional[str] = None,
):
    """Plot decision boundary on 2D projected data (e.g., t-SNE)."""
    h = 0.5
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="Set3")
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    for label in sorted(set(y)):
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[int(label)], label=CLASS_NAMES[int(label)], alpha=0.6, s=15)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    return fig


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build comparison DataFrame from {model_name: eval_result} dict."""
    rows = []
    for name, res in results.items():
        r = res["report"]
        rows.append({
            "Model": name,
            "Set": res["set_name"],
            "Accuracy": res["accuracy"],
            "Precision (macro)": r["macro avg"]["precision"],
            "Recall (macro)": r["macro avg"]["recall"],
            "F1 (macro)": r["macro avg"]["f1-score"],
        })
    return pd.DataFrame(rows)
