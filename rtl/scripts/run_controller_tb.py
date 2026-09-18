#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_lif_tb import find_simulator, simulator_version
from run_recurrent_tb import load_edges, write_graph_memories
from src.hardware.hardware_model import IntegerLIFReservoir


VECTOR_DIR = ROOT / "outputs" / "fpga" / "golden_vectors"
MODEL_FILE = ROOT / "outputs" / "runs" / "run_001" / "hardware_model.npz"
MEM_DIR = ROOT / "rtl" / "mem"
NEURONS = 64
TIMESTEPS = 20


def unsigned_hex(value: int, bits: int) -> str:
    return f"{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def write_mem(path: Path, values, bits: int) -> None:
    with path.open("w", encoding="ascii", newline="\n") as f:
        for value in values:
            f.write(unsigned_hex(int(value), bits) + "\n")


def vector_to_bits(values: np.ndarray) -> int:
    return sum(int(value) << index for index, value in enumerate(values))


def build_segments() -> tuple[list[str], dict[str, np.ndarray], list[int]]:
    with np.load(VECTOR_DIR / "golden_vectors.npz", allow_pickle=False) as stored:
        real_input_q = stored["input_quantized"].astype(np.int32)
        real_ids = [str(value) for value in stored["sample_ids"]]
        real_splits = [str(value) for value in stored["source_splits"]]

    with np.load(MODEL_FILE, allow_pickle=False) as model:
        reservoir = IntegerLIFReservoir(model["w_in"], model["w_res"])

    synthetic = np.asarray(
        [
            np.zeros(TIMESTEPS, dtype=np.float32),
            np.full(TIMESTEPS, 64 / 1024, dtype=np.float32),
            np.full(TIMESTEPS, 384 / 1024, dtype=np.float32),
            np.full(TIMESTEPS, 593 / 1024, dtype=np.float32),
            np.asarray([593 / 1024 if timestep % 2 == 0 else 0 for timestep in range(TIMESTEPS)], dtype=np.float32),
        ]
    )
    real_input = real_input_q.astype(np.float32) / 1024
    all_input = np.concatenate((synthetic, real_input), axis=0)
    trace = reservoir.transform_trace(all_input)

    expected_real = set(range(5, 5 + len(real_ids)))
    for real_index in expected_real:
        stored_index = real_index - 5
        assert np.array_equal(trace["input_quantized"][real_index], real_input_q[stored_index])

    names = [
        "synthetic_zero",
        "synthetic_low_sparse",
        "synthetic_mid",
        "synthetic_high",
        "synthetic_pulse",
    ] + real_ids
    # The first three segments deliberately exercise A, B, A reset isolation.
    sequence = [5, 9, 5, 0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12]
    sequence_names = [names[index] for index in sequence]

    selected = {}
    for field, values in trace.items():
        if isinstance(values, np.ndarray) and values.ndim >= 1 and values.shape[0] == all_input.shape[0]:
            selected[field] = values[sequence]
    selected["segment_names"] = np.asarray(sequence_names, dtype="U")
    return sequence_names, selected, sequence


def write_controller_memories(names: list[str], vectors: dict[str, np.ndarray]) -> int:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    write_mem(MEM_DIR / "controller_inputs.mem", vectors["input_quantized"].reshape(-1), 12)
    write_mem(MEM_DIR / "controller_expected_membrane_before.mem", vectors["membrane_before"].reshape(-1), 16)
    write_mem(MEM_DIR / "controller_expected_membrane_after.mem", vectors["membrane_after_update"].reshape(-1), 16)
    write_mem(MEM_DIR / "controller_expected_membrane_reset.mem", vectors["reset_membrane"].reshape(-1), 16)
    write_mem(MEM_DIR / "controller_expected_input.mem", vectors["input_contribution"].reshape(-1), 32)
    write_mem(MEM_DIR / "controller_expected_recurrent.mem", vectors["recurrent_contribution"].reshape(-1), 32)
    write_mem(MEM_DIR / "controller_expected_spikes.mem", vectors["threshold_result"].reshape(-1), 1)
    write_mem(MEM_DIR / "controller_expected_counts.mem", vectors["spike_count_state"].reshape(-1), 5)
    previous = []
    for segment in range(len(names)):
        for timestep in range(TIMESTEPS):
            if timestep == 0:
                previous.append(0)
            else:
                previous.append(vector_to_bits(vectors["threshold_result"][segment, timestep - 1]))
    write_mem(MEM_DIR / "controller_expected_previous_spikes.mem", previous, 64)
    write_mem(MEM_DIR / "controller_expected_final_counts.mem", vectors["spike_counts"].reshape(-1), 5)
    write_mem(MEM_DIR / "controller_segment_code.mem", list(range(len(names))), 8)
    return len(names) * TIMESTEPS


def compile_and_run(iverilog: str, vvp: str, checkpoints: int, segments: int) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="controller_tb_") as temp_dir:
        sim_path = Path(temp_dir) / "tb_reservoir_controller.vvp"
        compile_cmd = [
            iverilog,
            "-g2012",
            "-s",
            "tb_reservoir_controller",
            f"-Ptb_reservoir_controller.NUM_SEGMENTS={segments}",
            f"-Ptb_reservoir_controller.NUM_CHECKPOINTS={checkpoints}",
            "-o",
            str(sim_path),
            "rtl/src/fixed_point_pkg.sv",
            "rtl/src/lif_pe.sv",
            "rtl/src/sparse_recurrent_engine.sv",
            "rtl/src/reservoir_step.sv",
            "rtl/src/reservoir_controller.sv",
            "rtl/tb/tb_reservoir_controller.sv",
        ]
        compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
        if compiled.returncode != 0:
            return compiled.returncode, compiled.stdout + compiled.stderr
        executed = subprocess.run([vvp, str(sim_path)], cwd=ROOT, text=True, capture_output=True)
        return executed.returncode, executed.stdout + executed.stderr


def main() -> int:
    iverilog, vvp, simulator = find_simulator()
    if not iverilog or not vvp:
        print("SKIP: iverilog/vvp simulator not available; controller RTL was not executed")
        return 2

    # Reuse the Phase 2 graph validation and memory representation.
    edge_rows, _ = load_edges()
    write_graph_memories(edge_rows)
    names, vectors, _ = build_segments()
    checkpoints = write_controller_memories(names, vectors)
    if int(np.max(vectors["spike_count_state"])) > 20:
        raise RuntimeError("Golden spike count exceeded the valid 20-timestep range")

    print(f"Simulator: {simulator}; {simulator_version(iverilog)}")
    print(f"Segments: {len(names)}; checkpoints: {checkpoints}; neurons per checkpoint: {NEURONS}")
    print("Segment order: " + ", ".join(f"{index}={name}" for index, name in enumerate(names)))
    code, output = compile_and_run(iverilog, vvp, checkpoints, len(names))
    print(output, end="")
    if code != 0 or "PASS:" not in output:
        return code or 1
    print("Mismatches: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
