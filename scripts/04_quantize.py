#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import dataframe_to_xy, load_dataframe
from src.evaluation.metrics import evaluate_binary_classifier
from src.evaluation.plots import plot_metric_comparison
from src.hardware.fixed_point import FixedPointFormat
from src.hardware.hardware_model import (
    HardwareParams,
    IntegerLIFReservoir,
    integer_predict,
    integer_readout_diagnostics,
)
from src.hardware.quantization import quantize_symmetric
from src.model.readout import NoBiasRidgeReadout
from src.utils.io import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate bit-oriented hardware model without using test data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/run_001")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    run_dir = ROOT / args.run_dir
    float_model = np.load(run_dir / "float_model.npz")

    train_df = load_dataframe(ROOT / "data" / "splits" / "train.csv")
    val_df = load_dataframe(ROOT / "data" / "splits" / "val.csv")
    X_train, y_train = dataframe_to_xy(train_df, cfg["data"]["sequence_length"])
    X_val, y_val = dataframe_to_xy(val_df, cfg["data"]["sequence_length"])

    h = cfg["hardware"]
    hp = HardwareParams(
        input_format=FixedPointFormat(h["input_bits"], h["input_fractional_bits"], signed=True),
        membrane_bits=h["membrane_bits"],
        membrane_fractional_bits=h["membrane_fractional_bits"],
        recurrent_accumulator_bits=h["recurrent_accumulator_bits"],
        threshold=cfg["reservoir"]["threshold"],
        leak_shift=3,
        recurrent_shift=4,
        saturation=h["saturation"],
    )
    reservoir = IntegerLIFReservoir(float_model["w_in"], float_model["w_res"], hp)
    F_train, train_fixed_diagnostics = reservoir.transform(X_train, return_diagnostics=True)
    F_val, val_fixed_diagnostics = reservoir.transform(X_val, return_diagnostics=True)

    readout = NoBiasRidgeReadout(
        alpha=cfg["readout"]["alpha"],
        class_balanced_sample_weight=cfg["readout"]["class_balanced_sample_weight"],
    ).fit(F_train, y_train)
    q = quantize_symmetric(readout.weights, h["readout_weight_bits"])
    pred, score_int = integer_predict(F_val, q.values, h["accumulator_bits"])
    train_readout_diagnostics = integer_readout_diagnostics(F_train, q.values, h["accumulator_bits"])
    val_readout_diagnostics = integer_readout_diagnostics(F_val, q.values, h["accumulator_bits"])
    metrics = evaluate_binary_classifier(y_val, pred, score_int)

    # Float-readout-on-integer-features baseline for quantization-drop accounting.
    score_float = F_val @ readout.weights
    pred_float = (score_float >= 0).astype(np.int64)
    float_metrics = evaluate_binary_classifier(y_val, pred_float, score_float)
    drop = float_metrics["balanced_accuracy"] - metrics["balanced_accuracy"]

    np.savez_compressed(
        run_dir / "hardware_model.npz",
        w_in=float_model["w_in"],
        w_res=float_model["w_res"],
        w_out_float=readout.weights,
        w_out_int8=q.values,
        w_out_scale=np.asarray([q.scale], dtype=np.float64),
    )
    np.save(run_dir / "features_train_fixed.npy", F_train)
    np.save(run_dir / "features_val_fixed.npy", F_val)
    summary = {
        "fixed_point_validation_float_readout": float_metrics,
        "fixed_point_validation_int8_readout": metrics,
        "balanced_accuracy_quantization_drop": float(drop),
        "readout_quantization_scale": q.scale,
        "readout_weight_min": int(q.values.min()),
        "readout_weight_max": int(q.values.max()),
        "test_used": False,
        "fixed_point_diagnostics": {
            "train": train_fixed_diagnostics,
            "validation": val_fixed_diagnostics,
            "train_readout": train_readout_diagnostics,
            "validation_readout": val_readout_diagnostics,
        },
    }
    save_json(summary, run_dir / "quantization_summary.json")
    plot_metric_comparison(
        [
            {"name": "Fixed + float Wout", **float_metrics},
            {"name": "Fixed + INT8 Wout", **metrics},
        ],
        run_dir / "figures" / "quantization_balanced_accuracy.png",
    )
    print(f"INT8 validation balanced accuracy={metrics['balanced_accuracy']:.6f}; drop={drop:.6f}")


if __name__ == "__main__":
    main()
