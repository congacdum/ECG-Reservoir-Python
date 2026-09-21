#!/usr/bin/env python3
"""Deterministic train/validation multi-seed stability experiment."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from analysis_common import ROOT, ensure_analysis_dir, load_train_validation
sys.path.insert(0, str(ROOT))
from src.model.reservoir_numpy import NumpyLIFReservoir, ReservoirParams
from src.model.readout import NoBiasRidgeReadout


def main() -> int:
    x_train, y_train, x_val, y_val = load_train_validation()
    rows = []
    for seed in range(10):
        params = ReservoirParams(seed=seed)
        reservoir = NumpyLIFReservoir(params)
        train_features = reservoir.transform(x_train)
        val_features = reservoir.transform(x_val)
        readout = NoBiasRidgeReadout(alpha=100.0, class_balanced_sample_weight=True).fit(train_features, y_train)
        train_pred = readout.predict(train_features)
        val_pred = readout.predict(val_features)
        rows.append({
            "seed": seed, "edge_count": reservoir.recurrent_edges,
            "train_balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
            "val_balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
            "train_accuracy": accuracy_score(y_train, train_pred),
            "val_accuracy": accuracy_score(y_val, val_pred),
        })
    frame = pd.DataFrame(rows)
    out = ensure_analysis_dir()
    frame.to_csv(out / "reservoir_seed_stability.csv", index=False)
    summary = {
        "seed_count": len(frame), "seeds": list(range(10)), "test_used": False,
        "val_balanced_accuracy_mean": frame["val_balanced_accuracy"].mean(),
        "val_balanced_accuracy_std": frame["val_balanced_accuracy"].std(ddof=1),
        "val_balanced_accuracy_min": frame["val_balanced_accuracy"].min(),
        "val_balanced_accuracy_max": frame["val_balanced_accuracy"].max(),
    }
    (out / "reservoir_seed_stability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

