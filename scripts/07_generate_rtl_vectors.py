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
from src.hardware.fixed_point import FixedPointFormat
from src.hardware.golden_vectors import generate_golden_vectors, write_golden_vectors
from src.hardware.hardware_model import HardwareParams, IntegerLIFReservoir
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RTL golden vectors from the bit-exact fixed-point model."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/run_001")
    parser.add_argument("--output-dir", default="outputs/fpga/golden_vectors")
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.train_count < 0 or args.validation_count < 0:
        raise SystemExit("Vector counts must be non-negative")

    cfg = load_yaml(args.config)
    output_dir = ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --force explicitly.")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = np.load(ROOT / args.run_dir / "hardware_model.npz")
    train_df = load_dataframe(ROOT / "data" / "splits" / "train.csv").iloc[: args.train_count]
    val_df = load_dataframe(ROOT / "data" / "splits" / "val.csv").iloc[: args.validation_count]
    X_train, y_train = dataframe_to_xy(train_df, cfg["data"]["sequence_length"])
    X_val, y_val = dataframe_to_xy(val_df, cfg["data"]["sequence_length"])
    X = np.concatenate([X_train, X_val], axis=0)
    labels = np.concatenate([y_train, y_val], axis=0)
    sample_ids = [f"train_{i:04d}" for i in range(len(X_train))]
    sample_ids.extend(f"validation_{i:04d}" for i in range(len(X_val)))
    source_splits = ["train"] * len(X_train) + ["validation"] * len(X_val)

    h = cfg["hardware"]
    params = HardwareParams(
        input_format=FixedPointFormat(h["input_bits"], h["input_fractional_bits"], signed=True),
        membrane_bits=h["membrane_bits"],
        membrane_fractional_bits=h["membrane_fractional_bits"],
        recurrent_accumulator_bits=h["recurrent_accumulator_bits"],
        threshold=cfg["reservoir"]["threshold"],
        leak_shift=3,
        recurrent_shift=4,
        saturation=h["saturation"],
    )
    reservoir = IntegerLIFReservoir(model["w_in"], model["w_res"], params)
    vectors = generate_golden_vectors(
        reservoir,
        X,
        labels,
        model["w_out_int8"],
        sample_ids,
        source_splits,
        accumulator_bits=h["accumulator_bits"],
    )
    manifest = write_golden_vectors(
        output_dir,
        vectors,
        reservoir,
        accumulator_bits=h["accumulator_bits"],
        reservoir_seed=cfg["project"]["seed"],
    )
    shutil.copy2(ROOT / args.config, output_dir / "config.yaml")
    print(
        f"Generated {manifest['sample_count']} vectors from train/validation; "
        f"model_sha256={manifest['model_sha256']}"
    )


if __name__ == "__main__":
    main()
