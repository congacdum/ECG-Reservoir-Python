#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_lif_tb import find_simulator, simulator_version

GOLDEN_DIR = ROOT / "outputs" / "fpga" / "golden_vectors"
MEM_DIR = ROOT / "rtl" / "mem"
NEURONS = 64
TIMESTEPS = 20


def signed(raw: int, bits: int) -> int:
    raw &= (1 << bits) - 1
    return raw - (1 << bits) if raw & (1 << (bits - 1)) else raw


def hex_line(value: int, bits: int) -> str:
    return f"{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def write_mem(path: Path, values, bits: int) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for value in values:
            handle.write(hex_line(int(value), bits) + "\n")


def parse_rows(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        tokens = line.split()
        if tokens:
            rows.append([int(token, 16) for token in tokens])
    return rows


def pack_counts(values: list[int]) -> int:
    return sum(int(value) << (index * 5) for index, value in enumerate(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the integrated classifier RTL regression")
    parser.add_argument("--smoke", action="store_true", help="run the first four deterministic golden samples")
    args = parser.parse_args()

    iverilog, vvp, simulator = find_simulator()
    if not iverilog or not vvp:
        print("SKIP: iverilog/vvp simulator not available; classifier RTL was not executed")
        return 2

    inputs = parse_rows(GOLDEN_DIR / "inputs_q12.mem")
    counts = parse_rows(GOLDEN_DIR / "spike_counts_5bit.mem")
    score_tokens = (GOLDEN_DIR / "expected_scores_20bit.mem").read_text(encoding="ascii").split()
    class_tokens = (GOLDEN_DIR / "expected_classes.mem").read_text(encoding="ascii").split()
    if len(inputs) != 8 or any(len(row) != TIMESTEPS for row in inputs):
        raise RuntimeError("expected eight 20-timestep input vectors")
    if len(counts) != 8 or any(len(row) != NEURONS for row in counts):
        raise RuntimeError("expected eight 64-neuron count vectors")
    scores = [signed(int(token, 16), 20) for token in score_tokens]
    classes = [int(token, 2) for token in class_tokens]
    if args.smoke:
        inputs = inputs[:4]
        counts = counts[:4]
        scores = scores[:4]
        classes = classes[:4]

    MEM_DIR.mkdir(parents=True, exist_ok=True)
    write_mem(MEM_DIR / "classifier_inputs.mem", [value for row in inputs for value in row], 12)
    write_mem(MEM_DIR / "classifier_expected_counts.mem", [pack_counts(row) for row in counts], 320)
    write_mem(MEM_DIR / "classifier_expected_score.mem", scores, 20)
    write_mem(MEM_DIR / "classifier_expected_class.mem", classes, 1)

    with tempfile.TemporaryDirectory(prefix="classifier_tb_") as temp_dir:
        sim_path = Path(temp_dir) / "tb_ecg_classifier_core.vvp"
        compile_cmd = [
            iverilog, "-g2012", "-s", "tb_ecg_classifier_core",
            f"-Ptb_ecg_classifier_core.NUM_SAMPLES={len(inputs)}", "-o", str(sim_path),
            "rtl/src/fixed_point_pkg.sv",
            "rtl/src/lif_pe.sv",
            "rtl/src/sparse_recurrent_engine.sv",
            "rtl/src/reservoir_step.sv",
            "rtl/src/reservoir_controller.sv",
            "rtl/src/readout_mac.sv",
            "rtl/src/ecg_classifier_core.sv",
            "rtl/tb/tb_ecg_classifier_core.sv",
        ]
        compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
        if compiled.returncode != 0:
            print(compiled.stdout + compiled.stderr, end="")
            return compiled.returncode
        executed = subprocess.run([vvp, str(sim_path)], cwd=ROOT, text=True, capture_output=True)
        print(f"Simulator: {simulator}; {simulator_version(iverilog)}")
        print(f"Classifier samples: {len(inputs)}; timesteps per sample: {TIMESTEPS}")
        print(executed.stdout + executed.stderr, end="")
        if executed.returncode != 0 or "PASS:" not in executed.stdout:
            return executed.returncode or 1
    print("Mismatches: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
