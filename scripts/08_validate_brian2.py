#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import dataframe_to_xy, load_dataframe
from src.evaluation.metrics import evaluate_binary_classifier
from src.model.readout import NoBiasRidgeReadout
from src.model.reservoir_brian2 import transform_brian2
from src.utils.io import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Brian2 exploratory backend on train/validation only.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-run-dir", default="outputs/runs/run_001")
    parser.add_argument("--run-dir", default="outputs/runs/run_brian2_20260919")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")

    cfg = load_yaml(args.config)
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / args.config, run_dir / "config.yaml")
    model = np.load(ROOT / args.model_run_dir / "float_model.npz")

    train_df = load_dataframe(ROOT / "data" / "splits" / "train.csv")
    val_df = load_dataframe(ROOT / "data" / "splits" / "val.csv")
    X_train, y_train = dataframe_to_xy(train_df, cfg["data"]["sequence_length"])
    X_val, y_val = dataframe_to_xy(val_df, cfg["data"]["sequence_length"])

    start = time.perf_counter()
    F_train = transform_brian2(
        X_train, model["w_in"], model["w_res"], batch_size=args.batch_size
    )
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    F_val = transform_brian2(
        X_val, model["w_in"], model["w_res"], batch_size=args.batch_size
    )
    validation_seconds = time.perf_counter() - start

    readout = NoBiasRidgeReadout(
        alpha=cfg["readout"]["alpha"],
        class_balanced_sample_weight=cfg["readout"]["class_balanced_sample_weight"],
    ).fit(F_train, y_train)
    train_score = F_train @ readout.weights
    val_score = F_val @ readout.weights
    train_pred = (train_score >= cfg["readout"]["decision_threshold"]).astype(np.int64)
    val_pred = (val_score >= cfg["readout"]["decision_threshold"]).astype(np.int64)
    train_metrics = evaluate_binary_classifier(y_train, train_pred, train_score)
    val_metrics = evaluate_binary_classifier(y_val, val_pred, val_score)

    np.save(run_dir / "features_train_brian2.npy", F_train)
    np.save(run_dir / "features_val_brian2.npy", F_val)
    save_json(
        {
            "backend": "brian2_exploratory_reference",
            "batch_size": args.batch_size,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "train_nonzero_spike_count_rate": float(np.count_nonzero(F_train) / F_train.size),
            "validation_nonzero_spike_count_rate": float(np.count_nonzero(F_val) / F_val.size),
            "train": train_metrics,
            "validation": val_metrics,
            "readout_fit_intercept": False,
            "test_used_for_model_selection": False,
            "fpga_golden_model": "src/hardware/hardware_model.py",
            "audit": {
                "sequence_length": cfg["data"]["sequence_length"],
                "dt_ms": 1.0,
                "leak_tau_ms": 8.0,
                "input_scaling": "tau/integration_dt correction gives direct input current per Euler step",
                "input_weight_mapping": "one fixed w_in value per neuron",
                "threshold": cfg["reservoir"]["threshold"],
                "reset_value": 0.0,
                "refractory_period": "none",
                "recurrent_mapping": "w_res[destination, source] mapped source to destination",
                "recurrent_gain": cfg["reservoir"]["recurrent_gain_numerator"] / cfg["reservoir"]["recurrent_gain_denominator"],
                "synaptic_delay_ms": 0.0,
                "sample_state_isolation": "disjoint neuron blocks per batch",
                "units": "dimensionless membrane/weights; milliseconds for dt/tau",
                "deterministic": True,
                "remaining_difference": "recurrent event-driven floating-point accumulation can diverge sparsely from vectorized NumPy recurrence",
            },
            "known_difference": "Brian2 event-driven floating-point recurrence can diverge from discrete NumPy recurrence; Brian2 is not the FPGA golden model.",
        },
        run_dir / "brian2_validation_metrics.json",
    )
    print(
        f"Brian2 validation complete. BA={val_metrics['balanced_accuracy']:.6f}; "
        f"train_seconds={train_seconds:.3f}; validation_seconds={validation_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
