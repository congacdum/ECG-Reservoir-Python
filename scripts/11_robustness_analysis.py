#!/usr/bin/env python3
"""Measure locked-model validation robustness; no architecture tuning."""
from __future__ import annotations

import numpy as np
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from analysis_common import ROOT, ensure_analysis_dir, load_train_validation
sys.path.insert(0, str(ROOT))
from src.hardware.hardware_model import IntegerLIFReservoir, integer_readout_score
from src.model.reservoir_numpy import NumpyLIFReservoir, ReservoirParams


def metrics(y_true, score):
    pred = (score >= 0).astype(np.int64)
    return float(balanced_accuracy_score(y_true, pred)), float(accuracy_score(y_true, pred))


def main() -> int:
    _, _, x_val, y_val = load_train_validation()
    with np.load(ROOT / "outputs" / "runs" / "run_001" / "hardware_model.npz", allow_pickle=False) as model:
        w_in = model["w_in"]
        w_res = model["w_res"]
        w_float = model["w_out_float"]
        w_int = model["w_out_int8"]
    fixed = IntegerLIFReservoir(w_in, w_res)
    float_reservoir = NumpyLIFReservoir(ReservoirParams(seed=42), w_in=w_in, w_res=w_res)
    clean_fixed = integer_readout_score(fixed.transform(x_val), w_int)
    clean_float = float_reservoir.transform(x_val).astype(np.float64) @ w_float
    clean_fixed_ba, clean_fixed_acc = metrics(y_val, clean_fixed)
    clean_float_ba, clean_float_acc = metrics(y_val, clean_float)
    rows = []
    rng = np.random.default_rng(42)
    cases = []
    for value in np.arange(-0.05, 0.051, 0.01):
        cases.append(("dc_offset", float(np.round(value, 5)), x_val + value))
    for value in (0.8, 0.9, 1.0, 1.1, 1.2):
        cases.append(("gain", value, x_val * value))
    for sigma in (0.0, 0.0025, 0.005, 0.01, 0.02):
        noise = rng.normal(0.0, sigma, size=x_val.shape).astype(np.float32)
        cases.append(("gaussian_noise", sigma, x_val + noise))
    for perturbation, value, data in cases:
        fixed_score = integer_readout_score(fixed.transform(data), w_int)
        float_score = float_reservoir.transform(data).astype(np.float64) @ w_float
        fixed_ba, fixed_acc = metrics(y_val, fixed_score)
        float_ba, float_acc = metrics(y_val, float_score)
        rows.extend([
            {"model": "fixed_int8", "perturbation": perturbation, "value": value,
             "balanced_accuracy": fixed_ba, "accuracy": fixed_acc,
             "delta_from_clean": fixed_ba - clean_fixed_ba},
            {"model": "float_reference", "perturbation": perturbation, "value": value,
             "balanced_accuracy": float_ba, "accuracy": float_acc,
             "delta_from_clean": float_ba - clean_float_ba},
        ])
    out = ensure_analysis_dir()
    pd.DataFrame(rows).to_csv(out / "robustness_metrics.csv", index=False)
    summary = {
        "clean_fixed_int8_balanced_accuracy": clean_fixed_ba,
        "clean_float_balanced_accuracy": clean_float_ba,
        "clean_fixed_int8_accuracy": clean_fixed_acc,
        "clean_float_accuracy": clean_float_acc,
        "test_used": False,
        "interpretation": "sensitivity evidence only; no architecture was changed",
    }
    (out / "robustness_summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



