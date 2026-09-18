from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=np.float64)
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["pr_auc"] = float(average_precision_score(y_true, y_score))
        result["score_min"] = float(np.min(y_score))
        result["score_max"] = float(np.max(y_score))
    return result


def evaluate_model_health(
    train_metrics: dict,
    validation_metrics: dict,
    overfitting_gap_threshold: float = 0.05,
    underfitting_balanced_accuracy_threshold: float = 0.60,
) -> dict:
    """Summarize train/validation gap with conservative diagnostic warnings."""
    train_ba = float(train_metrics["balanced_accuracy"])
    validation_ba = float(validation_metrics["balanced_accuracy"])
    gap = train_ba - validation_ba
    overfitting = train_ba >= underfitting_balanced_accuracy_threshold and gap > overfitting_gap_threshold
    underfitting = (
        train_ba < underfitting_balanced_accuracy_threshold
        and validation_ba < underfitting_balanced_accuracy_threshold
    )
    return {
        "train_balanced_accuracy": train_ba,
        "validation_balanced_accuracy": validation_ba,
        "generalization_gap": float(gap),
        "overfitting_warning": bool(overfitting),
        "underfitting_warning": bool(underfitting),
        "overfitting_gap_threshold": float(overfitting_gap_threshold),
        "underfitting_balanced_accuracy_threshold": float(underfitting_balanced_accuracy_threshold),
    }
