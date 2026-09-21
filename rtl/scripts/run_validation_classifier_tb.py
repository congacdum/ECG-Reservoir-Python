#!/usr/bin/env python3
"""Run the integrated classifier on a representative validation-only subset."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))
from run_lif_tb import find_simulator, simulator_version
from src.hardware.hardware_model import IntegerLIFReservoir, integer_readout_score
from analysis_common import load_train_validation


def hex_line(value: int, bits: int) -> str:
    return f"{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def write_mem(path: Path, values, bits: int) -> None:
    path.write_text("".join(hex_line(value, bits) + "\n" for value in values), encoding="ascii")


def pack_counts(values: np.ndarray) -> int:
    return sum(int(value) << (index * 5) for index, value in enumerate(values))


def main() -> int:
    iverilog, vvp, simulator = find_simulator()
    if not iverilog or not vvp:
        print("SKIP: iverilog/vvp unavailable")
        return 2
    _, _, x_val, y_val = load_train_validation()
    count = min(64, len(x_val))
    x_val = x_val[:count]
    y_val = y_val[:count]
    with np.load(ROOT / "outputs" / "runs" / "run_001" / "hardware_model.npz", allow_pickle=False) as model:
        reservoir = IntegerLIFReservoir(model["w_in"], model["w_res"])
        w_out = model["w_out_int8"]
    trace = reservoir.transform_trace(x_val)
    counts = trace["spike_counts"]
    scores = integer_readout_score(counts, w_out)
    classes = (scores >= 0).astype(np.int64)
    mem_dir = ROOT / "rtl" / "mem"
    names = ["classifier_inputs.mem", "classifier_expected_counts.mem", "classifier_expected_score.mem", "classifier_expected_class.mem"]
    with tempfile.TemporaryDirectory(prefix="validation_classifier_mem_") as backup_dir:
        backup_dir = Path(backup_dir)
        for name in names:
            shutil.copy2(mem_dir / name, backup_dir / name)
        try:
            write_mem(mem_dir / names[0], trace["input_quantized"].reshape(-1), 12)
            write_mem(mem_dir / names[1], [pack_counts(row) for row in counts], 320)
            write_mem(mem_dir / names[2], scores, 20)
            write_mem(mem_dir / names[3], classes, 1)
            with tempfile.TemporaryDirectory(prefix="validation_classifier_tb_") as temp_dir:
                sim_path = Path(temp_dir) / "tb_ecg_classifier_validation.vvp"
                compile_cmd = [
                    iverilog, "-g2012", "-s", "tb_ecg_classifier_core",
                    f"-Ptb_ecg_classifier_core.NUM_SAMPLES={count}", "-o", str(sim_path),
                    "rtl/src/fixed_point_pkg.sv", "rtl/src/lif_pe.sv",
                    "rtl/src/sparse_recurrent_engine.sv", "rtl/src/reservoir_step.sv",
                    "rtl/src/reservoir_controller.sv", "rtl/src/readout_mac.sv",
                    "rtl/src/ecg_classifier_core.sv", "rtl/tb/tb_ecg_classifier_core.sv",
                ]
                compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
                if compiled.returncode != 0:
                    print(compiled.stdout + compiled.stderr, end="")
                    return compiled.returncode
                executed = subprocess.run([vvp, str(sim_path)], cwd=ROOT, text=True, capture_output=True)
                print(f"Simulator: {simulator}; {simulator_version(iverilog)}")
                print(f"Validation samples tested: {count}; test split read: no")
                print(executed.stdout + executed.stderr, end="")
                if executed.returncode != 0 or f"PASS: full classifier samples={count}" not in executed.stdout:
                    return executed.returncode or 1
        finally:
            for name in names:
                shutil.copy2(backup_dir / name, mem_dir / name)
    print("Validation E2E mismatches: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


