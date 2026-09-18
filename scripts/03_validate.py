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
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_score_distribution,
)
from src.model.reservoir_numpy import NumpyLIFReservoir
from src.utils.io import load_yaml, save_json
from src.model.factory import reservoir_params_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate float reference model on validation only.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/run_001")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    run_dir = ROOT / args.run_dir
    model = np.load(run_dir / "float_model.npz")
    val_df = load_dataframe(ROOT / "data" / "splits" / "val.csv")
    X, y = dataframe_to_xy(val_df, cfg["data"]["sequence_length"])
    reservoir = NumpyLIFReservoir(
        reservoir_params_from_config(cfg),
        w_in=model["w_in"],
        w_res=model["w_res"],
    )
    F = reservoir.transform(X)
    score = F @ model["w_out_float"]
    pred = (score >= 0).astype(np.int64)
    metrics = evaluate_binary_classifier(y, pred, score)
    save_json(metrics, run_dir / "validation_float_metrics.json")
    fig = run_dir / "figures"
    plot_confusion_matrix(y, pred, fig / "validation_float_confusion_matrix.png")
    plot_roc_curve(y, score, fig / "validation_float_roc_curve.png")
    plot_precision_recall_curve(y, score, fig / "validation_float_pr_curve.png")
    plot_score_distribution(y, score, fig / "validation_float_score_distribution.png")
    print(f"Validation balanced accuracy={metrics['balanced_accuracy']:.6f}")


if __name__ == "__main__":
    main()
