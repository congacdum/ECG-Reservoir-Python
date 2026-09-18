#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import dataframe_to_xy, load_dataframe
from src.data.checks import audit_split_files
from src.evaluation.metrics import evaluate_binary_classifier, evaluate_model_health
from src.evaluation.plots import plot_spike_count_distribution, plot_spike_raster
from src.model.readout import NoBiasRidgeReadout
from src.model.reservoir_numpy import NumpyLIFReservoir
from src.model.factory import reservoir_params_from_config
from src.utils.io import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train no-bias readout on fixed NumPy LIF reservoir features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/run_001")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    shutil.copy2(ROOT / args.config, run_dir / "config.yaml")

    split_paths = {name: ROOT / "data" / "splits" / f"{name}.csv" for name in ("train", "val")}
    # Training is intentionally blind to the held-out final split. Full split audit belongs to 01_prepare_data.py.
    audit_split_files(split_paths, cfg["data"]["sequence_length"])
    train_df = load_dataframe(split_paths["train"])
    val_df = load_dataframe(split_paths["val"])
    X_train, y_train = dataframe_to_xy(train_df, cfg["data"]["sequence_length"])
    X_val, y_val = dataframe_to_xy(val_df, cfg["data"]["sequence_length"])

    reservoir = NumpyLIFReservoir(reservoir_params_from_config(cfg))
    F_train = reservoir.transform(X_train)
    F_val, spikes_val = reservoir.transform(X_val[: min(64, len(X_val))], return_spikes=True)
    # Full validation features (the subset above is only for plotting raster)
    F_val_full = reservoir.transform(X_val)

    readout = NoBiasRidgeReadout(
        alpha=cfg["readout"]["alpha"],
        class_balanced_sample_weight=cfg["readout"]["class_balanced_sample_weight"],
    ).fit(F_train, y_train)
    train_score = F_train @ readout.weights
    train_pred = (train_score >= cfg["readout"]["decision_threshold"]).astype(np.int64)
    train_metrics = evaluate_binary_classifier(y_train, train_pred, train_score)
    val_score = F_val_full @ readout.weights
    val_pred = (val_score >= cfg["readout"]["decision_threshold"]).astype(np.int64)
    val_metrics = evaluate_binary_classifier(y_val, val_pred, val_score)
    diagnostic_cfg = cfg.get("diagnostics", {})
    model_health = evaluate_model_health(
        train_metrics,
        val_metrics,
        overfitting_gap_threshold=diagnostic_cfg.get("overfitting_gap_threshold", 0.05),
        underfitting_balanced_accuracy_threshold=diagnostic_cfg.get(
            "underfitting_balanced_accuracy_threshold", 0.60
        ),
    )

    np.savez_compressed(
        run_dir / "float_model.npz",
        w_in=reservoir.w_in,
        w_res=reservoir.w_res,
        w_out_float=readout.weights,
    )
    np.save(run_dir / "features_train_float.npy", F_train)
    np.save(run_dir / "features_val_float.npy", F_val_full)
    save_json(
        {
            "backend": "numpy_float_reference",
            "reservoir_edges": reservoir.recurrent_edges,
            "train": train_metrics,
            "validation": val_metrics,
            "model_health": model_health,
            "test_used_for_model_selection": False,
        },
        run_dir / "train_summary.json",
    )
    plot_spike_count_distribution(F_train, run_dir / "figures" / "train_spike_count_distribution.png")
    plot_spike_raster(spikes_val, run_dir / "figures" / "validation_spike_raster_example.png")
    print(
        "Training complete. "
        f"Train balanced accuracy={train_metrics['balanced_accuracy']:.6f}; "
        f"validation balanced accuracy={val_metrics['balanced_accuracy']:.6f}; "
        f"gap={model_health['generalization_gap']:.6f}"
    )


if __name__ == "__main__":
    main()
