#!/usr/bin/env python3
"""Train/validation-only recurrence, rank, and reservoir-size ablations."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from analysis_common import ROOT, ensure_analysis_dir, load_train_validation
sys.path.insert(0, str(ROOT))
from src.hardware.hardware_model import IntegerLIFReservoir
from src.model.readout import NoBiasRidgeReadout


def evaluate_readout(train_features, y_train, val_features, y_val):
    readout = NoBiasRidgeReadout(alpha=100.0, class_balanced_sample_weight=True).fit(train_features, y_train)
    train_score = readout.predict_score(train_features)
    val_score = readout.predict_score(val_features)
    train_pred = (train_score >= 0).astype(np.int64)
    val_pred = (val_score >= 0).astype(np.int64)
    return {
        "train_balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
        "val_balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
        "train_accuracy": accuracy_score(y_train, train_pred),
        "val_accuracy": accuracy_score(y_val, val_pred),
    }


def main() -> int:
    x_train, y_train, x_val, y_val = load_train_validation()
    run_dir = ROOT / "outputs" / "runs" / "run_001"
    with np.load(run_dir / "hardware_model.npz", allow_pickle=False) as model:
        w_in = model["w_in"]
        w_res = model["w_res"]
    out = ensure_analysis_dir()

    locked_train = np.load(run_dir / "features_train_fixed.npy")
    locked_val = np.load(run_dir / "features_val_fixed.npy")
    off_reservoir = IntegerLIFReservoir(w_in, np.zeros_like(w_res))
    off_train = off_reservoir.transform(x_train)
    off_val = off_reservoir.transform(x_val)
    off_metrics = evaluate_readout(off_train, y_train, off_val, y_val)
    recurrence_rows = [
        {"recurrence": "ON_locked_artifact", **evaluate_readout(locked_train, y_train, locked_val, y_val)},
        {"recurrence": "OFF_Wres_zero", **off_metrics},
    ]
    recurrence_frame = pd.DataFrame(recurrence_rows)
    recurrence_frame["delta_vs_off_val_BA"] = recurrence_frame["val_balanced_accuracy"] - off_metrics["val_balanced_accuracy"]
    recurrence_frame.to_csv(out / "recurrence_ablation.csv", index=False)

    centered = locked_train.astype(np.float64) - locked_train.astype(np.float64).mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular_values**2
    explained = variance / variance.sum()
    cumulative = np.cumsum(explained)
    rank_rows = []
    for target in (0.90, 0.95, 0.99):
        rank_rows.append({"target_explained_variance": target, "components": int(np.searchsorted(cumulative, target) + 1)})
    pd.DataFrame(rank_rows).to_csv(out / "reservoir_svd.csv", index=False)
    pd.DataFrame({
        "component": np.arange(1, len(explained) + 1),
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumulative,
    }).to_csv(out / "reservoir_svd_components.csv", index=False)

    size_rows = []
    for neurons in (8, 16, 32, 64):
        reservoir = IntegerLIFReservoir(w_in[:neurons], w_res[:neurons, :neurons])
        train_features = reservoir.transform(x_train)
        val_features = reservoir.transform(x_val)
        row = evaluate_readout(train_features, y_train, val_features, y_val)
        row.update({
            "neurons": neurons,
            "edge_count": int(np.count_nonzero(w_res[:neurons, :neurons])),
            "feature_dimension_95pct": int(rank_rows[1]["components"]) if neurons == 64 else "",
            "method": "prefix subset of locked graph; exploratory only",
        })
        size_rows.append(row)
    pd.DataFrame(size_rows).to_csv(out / "reservoir_size_sweep.csv", index=False)

    summary = {
        "test_used": False,
        "recurrence_ablation": recurrence_rows,
        "svd_targets": rank_rows,
        "size_sweep_method": "prefix subset of locked graph, not used to change locked N",
    }
    (out / "ablation_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(recurrence_frame.to_string(index=False))
    print(pd.DataFrame(rank_rows).to_string(index=False))
    print(pd.DataFrame(size_rows)[["neurons", "edge_count", "val_balanced_accuracy", "val_accuracy"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




