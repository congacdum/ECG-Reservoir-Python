#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hardware.export_fpga import export_hardware_artifacts
from src.utils.io import load_json, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Export locked model weights/config for FPGA RTL work.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/run_001")
    parser.add_argument("--output-dir", default="outputs/fpga/weights")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    model = np.load(ROOT / args.run_dir / "hardware_model.npz")
    validation = load_json(ROOT / args.run_dir / "quantization_summary.json")
    metadata = {
        "project": cfg["project"]["name"],
        "seed": cfg["project"]["seed"],
        "bias": False,
        "input_bits": cfg["hardware"]["input_bits"],
        "input_fractional_bits": cfg["hardware"]["input_fractional_bits"],
        "membrane_bits": cfg["hardware"]["membrane_bits"],
        "spike_count_bits": cfg["hardware"]["spike_count_bits"],
        "readout_weight_bits": cfg["hardware"]["readout_weight_bits"],
        "accumulator_bits": cfg["hardware"]["accumulator_bits"],
        "validation_balanced_accuracy": validation["fixed_point_validation_int8_readout"]["balanced_accuracy"],
        "readout_scale": float(model["w_out_scale"][0]),
    }
    export_hardware_artifacts(
        ROOT / args.output_dir,
        model["w_in"],
        model["w_res"],
        model["w_out_int8"],
        metadata,
    )
    out_cfg = ROOT / "outputs" / "fpga" / "config"
    out_cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / args.config, out_cfg / "locked_model_config.yaml")
    print(f"FPGA artifacts exported to {ROOT / args.output_dir}")


if __name__ == "__main__":
    main()
