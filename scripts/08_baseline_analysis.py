#!/usr/bin/env python3
"""Compare simple train/validation baselines without touching the sealed test set."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from analysis_common import ROOT, balanced_sample_weights, ensure_analysis_dir, feature_sets, load_train_validation, score_metrics


def run_model(name, model, x_train, y_train, x_val, y_val, sample_weight=False):
    fit_kwargs = {"sample_weight": balanced_sample_weights(y_train)} if sample_weight else {}
    model.fit(x_train, y_train, **fit_kwargs)
    train_score = model.decision_function(x_train) if hasattr(model, "decision_function") else model.predict_proba(x_train)[:, 1] - 0.5
    val_score = model.decision_function(x_val) if hasattr(model, "decision_function") else model.predict_proba(x_val)[:, 1] - 0.5
    train_metrics = score_metrics(y_train, train_score)
    val_metrics = score_metrics(y_val, val_score)
    return {
        "model": name, "feature_set": "",
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"],
        "train_roc_auc": train_metrics["roc_auc"], "val_roc_auc": val_metrics["roc_auc"],
        "notes": "class-balanced weights" if sample_weight else "default estimator weighting",
    }


def locked_reservoir_rows(y_train, y_val):
    run_dir = ROOT / "outputs" / "runs" / "run_001"
    with np.load(run_dir / "float_model.npz", allow_pickle=False) as model:
        w_float = model["w_out_float"]
    with np.load(run_dir / "hardware_model.npz", allow_pickle=False) as model:
        w_int = model["w_out_int8"]
    rows = []
    for model_name, weight, suffix in [
        ("reservoir_float_spike_count", w_float, "float"),
        ("reservoir_fixed_int8_spike_count", w_int, "fixed"),
    ]:
        train_features = np.load(run_dir / f"features_train_{suffix}.npy")
        val_features = np.load(run_dir / f"features_val_{suffix}.npy")
        train_score = train_features.astype(np.float64) @ np.asarray(weight, dtype=np.float64)
        val_score = val_features.astype(np.float64) @ np.asarray(weight, dtype=np.float64)
        train_metrics = score_metrics(y_train, train_score)
        val_metrics = score_metrics(y_val, val_score)
        rows.append({
            "model": model_name, "feature_set": "64 spike-count features",
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"],
            "train_roc_auc": train_metrics["roc_auc"], "val_roc_auc": val_metrics["roc_auc"],
            "notes": "locked artifact; no test data used",
        })
    return rows


def main() -> int:
    x_train, y_train, x_val, y_val = load_train_validation()
    rows = []
    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"), False),
        ("RidgeClassifier", RidgeClassifier(class_weight="balanced"), False),
        ("LinearSVC", LinearSVC(class_weight="balanced", random_state=42, max_iter=5000), False),
        ("DecisionTree_depth5", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42), False),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=42), True),
    ]
    for feature_name in ("RAW20", "RAW20_DIFF19"):
        train_set = feature_sets(x_train)[feature_name]
        val_set = feature_sets(x_val)[feature_name]
        for model_name, model, weighted_fit in models:
            row = run_model(model_name, model, train_set, y_train, val_set, y_val, weighted_fit)
            row["feature_set"] = feature_name
            rows.append(row)
    rows.extend(locked_reservoir_rows(y_train, y_val))
    output_dir = ensure_analysis_dir()
    metrics_path = output_dir / "baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    frame = pd.DataFrame(rows).sort_values("val_balanced_accuracy", ascending=False)
    best = frame.iloc[0]
    summary = {"metric": "validation_balanced_accuracy", "best_row": best.to_dict(),
               "reservoir_rows": [row for row in rows if row["model"].startswith("reservoir_")], "test_used": False}
    (output_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    lines = [
        "# Baseline analysis", "",
        "Đây là phân tích train/validation-only; final test không được đọc.", "",
        f"Best validation Balanced Accuracy: {best['val_balanced_accuracy']:.6f} ({best['model']} / {best['feature_set']}).", "",
        "RAW20 và RAW20_DIFF19 được đánh giá riêng. Linear/nonlinear baselines được đặt cạnh locked reservoir.",
        "Đây là evidence analysis, không phải model-selection trên final test.",
    ]
    (output_dir / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote baseline_metrics.csv")
    print(frame[["model", "feature_set", "val_balanced_accuracy", "val_accuracy", "val_roc_auc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



