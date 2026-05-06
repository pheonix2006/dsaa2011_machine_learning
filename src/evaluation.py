"""
Model evaluation utilities for Student Dropout and Academic Success dataset.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve, GridSearchCV
from sklearn.preprocessing import label_binarize


CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
CLASSES = [0, 1, 2]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate accuracy, precision, recall, F1 (macro and weighted)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_roc_curves(
    models: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot one-vs-rest macro ROC curves and attach per-class AUC values to the figure."""
    y_bin = label_binarize(y, classes=CLASSES)
    mean_fpr = np.linspace(0, 1, 100)
    auc_dict: dict[str, dict[str, float]] = {}

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)
        else:
            y_score = model.decision_function(X)

        class_aucs = {}
        interp_tprs = []
        for i, class_id in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            class_auc = auc(fpr, tpr)
            class_aucs[CLASS_NAMES[class_id]] = float(class_auc)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        macro_auc = auc(mean_fpr, mean_tpr)
        class_aucs["macro"] = float(macro_auc)
        auc_dict[name] = class_aucs
        ax.plot(mean_fpr, mean_tpr, linewidth=4, label=f"{name} (macro AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)
    ax.set_title("Macro-average ROC Curves", fontweight="bold", fontsize=24, pad=10)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower right", fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    fig.auc_dict = auc_dict
    return fig

def cross_validate_model(model, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "f1_macro"):
    """Run K-fold cross-validation and return mean/std of scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores,
        "cv": cv,
        "scoring": scoring,
    }


def plot_validation_curve(model, X: np.ndarray, y: np.ndarray, param_name: str, param_range: list, title: str = "Validation Curve", scoring: str = "f1_macro", cv: int = 5, save_path: Optional[str] = None):
    """Plot validation curve for a hyperparameter."""
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(range(len(param_range)), train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(range(len(param_range)), test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    ax.plot(range(len(param_range)), train_mean, "o-", color="blue", label="Training score")
    ax.plot(range(len(param_range)), test_mean, "o-", color="orange", label="Validation score")
    ax.set_xticks(range(len(param_range)))
    ax.set_xticklabels([str(v) for v in param_range])
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def plot_learning_curve(model, X: np.ndarray, y: np.ndarray, title: str = "Learning Curve", cv: int = 5, scoring: str = "f1_macro", save_path: Optional[str] = None):
    """Plot learning curve (training size vs score)."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="Cross-validation score")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(scoring)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def metrics_comparison_table(results: dict[str, dict]):
    """Build a metrics comparison DataFrame from {model_name: metrics_dict}."""
    rows = []
    for name, metrics in results.items():
        rows.append({"Model": name, **metrics})
    return pd.DataFrame(rows)


def tune_model(model, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, scoring: str = "f1_macro"):
    """Tune model hyperparameters via GridSearchCV."""
    gs = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    return {
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score": gs.best_score_,
        "cv_results": gs.cv_results_,
    }
