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
from src.hardware.fixed_point import FixedPointFormat
from src.hardware.hardware_model import HardwareParams, IntegerLIFReservoir, integer_predict
from src.utils.io import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="FINAL sealed-test evaluation. No tuning is allowed after inspecting this output.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/run_001")
    parser.add_argument("--confirm-final-test", action="store_true", help="Required explicit acknowledgement.")
    parser.add_argument("--force", action="store_true", help="Re-run even if final metrics already exist.")
    args = parser.parse_args()
    if not args.confirm_final_test:
        raise SystemExit("Refusing to open test set. Re-run with --confirm-final-test only after architecture is locked.")

    cfg = load_yaml(args.config)
    run_dir = ROOT / args.run_dir
    out_metrics = run_dir / "test_final_metrics.json"
    if out_metrics.exists() and not args.force:
        raise SystemExit(f"Final test metrics already exist at {out_metrics}. Refusing repeat evaluation without --force.")

    model = np.load(run_dir / "hardware_model.npz")
    test_df = load_dataframe(ROOT / "data" / "splits" / "test.csv")
    X, y = dataframe_to_xy(test_df, cfg["data"]["sequence_length"])
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
    reservoir = IntegerLIFReservoir(model["w_in"], model["w_res"], hp)
    F = reservoir.transform(X)
    pred, score = integer_predict(F, model["w_out_int8"], h["accumulator_bits"])
    metrics = evaluate_binary_classifier(y, pred, score)
    save_json(
        {
            **metrics,
            "status": "FINAL_TEST_AFTER_ARCHITECTURE_LOCK",
            "warning": "Do not tune architecture/hyperparameters using this result.",
        },
        out_metrics,
    )
    fig = run_dir / "figures"
    plot_confusion_matrix(y, pred, fig / "test_final_confusion_matrix.png")
    plot_roc_curve(y, score, fig / "test_final_roc_curve.png")
    plot_precision_recall_curve(y, score, fig / "test_final_pr_curve.png")
    plot_score_distribution(y, score, fig / "test_final_score_distribution.png")
    print(f"FINAL test balanced accuracy={metrics['balanced_accuracy']:.6f}")


if __name__ == "__main__":
    main()
