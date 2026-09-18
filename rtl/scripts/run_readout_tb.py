#!/usr/bin/env python3
from __future__ import annotations

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
WEIGHT_FILE = ROOT / "outputs" / "fpga" / "weights" / "w_out_int8.mem"
MEM_DIR = ROOT / "rtl" / "mem"
NEURONS = 64
COUNT_BITS = 5
ACC_BITS = 20


def signed(raw: int, bits: int) -> int:
    raw &= (1 << bits) - 1
    return raw - (1 << bits) if raw & (1 << (bits - 1)) else raw


def hex_line(value: int, bits: int) -> str:
    return f"{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def write_mem(path: Path, values, bits: int) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for value in values:
            handle.write(hex_line(int(value), bits) + "\n")


def parse_token_lines(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        tokens = line.split()
        if tokens:
            rows.append([int(token, 16) for token in tokens])
    return rows


def pack_counts(counts: list[int]) -> int:
    return sum(int(value) << (index * COUNT_BITS) for index, value in enumerate(counts))


def build_vectors() -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    weights = [signed(int(token, 16), 8) for token in WEIGHT_FILE.read_text(encoding="ascii").split()]
    if len(weights) != NEURONS:
        raise RuntimeError(f"expected {NEURONS} readout weights, found {len(weights)}")

    zeros = [0] * NEURONS
    one_positive = zeros.copy()
    one_positive[1] = 1
    one_negative = zeros.copy()
    one_negative[0] = 1
    exact_zero = zeros.copy()
    exact_zero[0] = 2
    exact_zero[2] = 3  # 2*(-3) + 3*(+2) == 0
    mixed = [((index * 7 + 3) % 21) for index in range(NEURONS)]
    all_twenty = [20] * NEURONS

    golden_rows = parse_token_lines(GOLDEN_DIR / "spike_counts_5bit.mem")
    if len(golden_rows) != 8 or any(len(row) != NEURONS for row in golden_rows):
        raise RuntimeError("golden spike-count memory is not eight 64-neuron vectors")

    vectors = [zeros, one_positive, one_negative, exact_zero, mixed, all_twenty] + golden_rows
    packed = [pack_counts(vector) for vector in vectors]
    expected_weights = []
    expected_products = []
    expected_before = []
    expected_after = []
    expected_scores = []
    expected_classes = []
    for vector in vectors:
        accumulator = 0
        for count, weight in zip(vector, weights):
            expected_weights.append(weight)
            product = int(count) * int(weight)
            expected_products.append(product)
            expected_before.append(accumulator)
            accumulator += product
            expected_after.append(accumulator)
        expected_scores.append(accumulator)
        expected_classes.append(int(accumulator >= 0))
    return packed, weights, expected_weights, expected_products, expected_before, expected_after, expected_scores, expected_classes


def compile_and_run(iverilog: str, vvp: str, vector_count: int) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="readout_tb_") as temp_dir:
        sim_path = Path(temp_dir) / "tb_readout_mac.vvp"
        compile_cmd = [
            iverilog, "-g2012", "-s", "tb_readout_mac",
            f"-Ptb_readout_mac.NUM_VECTORS={vector_count}", "-o", str(sim_path),
            "rtl/src/readout_mac.sv", "rtl/tb/tb_readout_mac.sv",
        ]
        compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
        if compiled.returncode != 0:
            return compiled.returncode, compiled.stdout + compiled.stderr
        executed = subprocess.run([vvp, str(sim_path)], cwd=ROOT, text=True, capture_output=True)
        return executed.returncode, executed.stdout + executed.stderr


def main() -> int:
    iverilog, vvp, simulator = find_simulator()
    if not iverilog or not vvp:
        print("SKIP: iverilog/vvp simulator not available; readout RTL was not executed")
        return 2

    (packed, weights, expected_weights, expected_products, expected_before,
     expected_after, expected_scores, expected_classes) = build_vectors()
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    write_mem(MEM_DIR / "readout_counts.mem", packed, NEURONS * COUNT_BITS)
    write_mem(MEM_DIR / "readout_weights.mem", expected_weights, 8)
    write_mem(MEM_DIR / "readout_expected_product.mem", expected_products, 32)
    write_mem(MEM_DIR / "readout_expected_acc_before.mem", expected_before, ACC_BITS)
    write_mem(MEM_DIR / "readout_expected_acc_after.mem", expected_after, ACC_BITS)
    write_mem(MEM_DIR / "readout_expected_score.mem", expected_scores, ACC_BITS)
    write_mem(MEM_DIR / "readout_expected_class.mem", expected_classes, 1)

    print(f"Simulator: {simulator}; {simulator_version(iverilog)}")
    print(f"Readout vectors: {len(packed)}; neurons per vector: {NEURONS}")
    print("Expected scores: " + ", ".join(str(value) for value in expected_scores))
    code, output = compile_and_run(iverilog, vvp, len(packed))
    print(output, end="")
    if code != 0 or "PASS:" not in output:
        return code or 1
    print("Mismatches: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
