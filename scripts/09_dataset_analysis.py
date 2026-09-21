#!/usr/bin/env python3
"""Investigate train/validation waveform artifacts without touching test.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from analysis_common import ensure_analysis_dir, feature_sets, load_train_validation


def descriptors(x: np.ndarray) -> pd.DataFrame:
    diff = np.diff(x, axis=1)
    return pd.DataFrame({
        "mean": x.mean(axis=1), "median": np.median(x, axis=1), "std": x.std(axis=1),
        "min": x.min(axis=1), "max": x.max(axis=1), "range": x.max(axis=1) - x.min(axis=1),
        "argmax": x.argmax(axis=1), "argmin": x.argmin(axis=1),
        "max_abs_diff": np.abs(diff).max(axis=1), "mean_abs_diff": np.abs(diff).mean(axis=1),
        "sum_abs_diff": np.abs(diff).sum(axis=1), "energy": np.square(x).sum(axis=1),
        "number_of_local_maxima": ((x[:, 1:-1] > x[:, :-2]) & (x[:, 1:-1] >= x[:, 2:])).sum(axis=1),
        "number_of_local_minima": ((x[:, 1:-1] < x[:, :-2]) & (x[:, 1:-1] <= x[:, 2:])).sum(axis=1),
    })


def best_threshold(train_values, y_train, val_values, y_val):
    best = None
    for direction in (1, -1):
        for threshold in np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, 257))):
            train_pred = (direction * train_values >= direction * threshold).astype(np.int64)
            val_pred = (direction * val_values >= direction * threshold).astype(np.int64)
            row = {
                "direction": direction, "threshold": float(threshold),
                "train_balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
                "val_balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
                "val_accuracy": accuracy_score(y_val, val_pred),
            }
            if best is None or row["train_balanced_accuracy"] > best["train_balanced_accuracy"]:
                best = row
    return best


def main() -> int:
    x_train, y_train, x_val, y_val = load_train_validation()
    out = ensure_analysis_dir()
    train_desc = descriptors(x_train).assign(split="train", label=y_train)
    val_desc = descriptors(x_val).assign(split="validation", label=y_val)
    pd.concat([train_desc, val_desc], ignore_index=True).to_csv(out / "window_statistics.csv", index=False)

    summary_rows = []
    descriptor_names = list(descriptors(x_train).columns)
    for split_name, frame in [("train", train_desc), ("validation", val_desc)]:
        for label, group in frame.groupby("label"):
            for feature in descriptor_names:
                values = group[feature]
                summary_rows.append({
                    "split": split_name, "label": int(label), "feature": feature,
                    "mean": values.mean(), "median": values.median(), "std": values.std(ddof=0),
                    "q25": values.quantile(0.25), "q75": values.quantile(0.75),
                    "min": values.min(), "max": values.max(),
                })
    pd.DataFrame(summary_rows).to_csv(out / "class_feature_statistics.csv", index=False)

    rule_rows = []
    for feature in ("range", "std", "max_abs_diff", "energy"):
        rule_rows.append({"feature": feature, **best_threshold(
            train_desc[feature].to_numpy(), y_train, val_desc[feature].to_numpy(), y_val
        )})
    pd.DataFrame(rule_rows).to_csv(out / "simple_rule_metrics.csv", index=False)

    importance_rows = []
    for feature_name in ("RAW20", "RAW20_DIFF19"):
        model = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
        train_features = feature_sets(x_train)[feature_name]
        model.fit(train_features, y_train)
        for index, importance in enumerate(model.feature_importances_):
            name = f"x{index}" if index < 20 else f"diff{index - 19}"
            importance_rows.append({
                "feature_set": feature_name, "feature": name,
                "tree_importance": importance,
            })
    pd.DataFrame(importance_rows).sort_values("tree_importance", ascending=False).to_csv(
        out / "baseline_feature_importance.csv", index=False
    )

    provenance = """# Dataset provenance investigation

Phạm vi: chỉ train và validation; test split không được đọc.

## Quan sát xác minh được

- File gốc chứa value_sequence, finalLabel, original_index, original_len, generated_window_id và new_len.
- value_sequence là chuỗi 20 giá trị được dùng làm input.
- finalLabel đã có sẵn trong dataset và được copy vào split artifacts.
- Tên file và metadata có mô tả single_peak_hardneg, nhưng repository không cung cấp đầy đủ source provenance bên ngoài hoặc định nghĩa lâm sàng của nhãn.

## Kết luận

The exact semantic/generation provenance of finalLabel could not be established from the repository. Kết quả baseline rất mạnh của tree/HistGradientBoosting là evidence cần điều tra thêm artifact/morphology; chưa đủ bằng chứng để gọi đây là data leakage.
"""
    (out / "dataset_provenance.md").write_text(provenance, encoding="utf-8")
    print("Wrote dataset analysis artifacts")
    print(pd.DataFrame(rule_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


