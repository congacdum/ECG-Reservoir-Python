#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from run_lif_tb import find_simulator, simulator_version


ROOT = Path(__file__).resolve().parents[2]
MEM_DIR = ROOT / "rtl" / "mem"
VECTOR_DIR = ROOT / "outputs" / "fpga" / "golden_vectors"
EDGE_FILE = ROOT / "outputs" / "fpga" / "weights" / "w_res_edges.csv"
NEURONS = 64
RECURRENT_UNIT = 1 << 6
RECURRENT_BITS = 16


def unsigned_hex(value: int, bits: int) -> str:
    return f"{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def write_mem(path: Path, values, bits: int) -> None:
    with path.open("w", encoding="ascii", newline="\n") as f:
        for value in values:
            f.write(unsigned_hex(int(value), bits) + "\n")


def load_edges() -> tuple[list[tuple[int, int, int]], dict[int, list[tuple[int, int]]]]:
    with EDGE_FILE.open(newline="", encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    edges = []
    incoming = {destination: [] for destination in range(NEURONS)}
    seen = set()
    for row in rows:
        source = int(row["source"])
        destination = int(row["destination"])
        sign = 1 if int(row["sign"]) == 0 else -1
        if not (0 <= source < NEURONS and 0 <= destination < NEURONS):
            raise RuntimeError(f"Invalid edge index: {row}")
        if source == destination:
            raise RuntimeError(f"Self-edge in frozen graph: {row}")
        if (source, destination) in seen:
            raise RuntimeError(f"Duplicate edge in frozen graph: {row}")
        seen.add((source, destination))
        edges.append((source, destination, sign))
        incoming[destination].append((source, sign))

    model = np.load(ROOT / "outputs" / "runs" / "run_001" / "hardware_model.npz", allow_pickle=False)
    w_res = np.asarray(model["w_res"], dtype=np.int8)
    reconstructed = np.zeros((NEURONS, NEURONS), dtype=np.int8)
    for source, destination, sign in edges:
        reconstructed[destination, source] = sign
    if not np.array_equal(reconstructed, w_res):
        raise RuntimeError("Exported edge orientation/content does not match frozen w_res and spikes @ w_res.T")
    return edges, incoming


def recurrent_values(spike_vector: int, destination: int, incoming) -> tuple[int, int, int, int]:
    active_edges = [(source, sign) for source, sign in incoming[destination] if (spike_vector >> source) & 1]
    edge_sum = sum(sign for _, sign in active_edges)
    scaled_raw = edge_sum * RECURRENT_UNIT
    scaled = min((1 << (RECURRENT_BITS - 1)) - 1, max(-(1 << (RECURRENT_BITS - 1)), scaled_raw))
    edge_mask = sum(1 << source for source, _ in active_edges)
    return edge_sum, scaled_raw, scaled, edge_mask


def build_unit_vectors(incoming):
    vectors = []

    def add(spike_vector: int, destination: int) -> None:
        edge_sum, scaled_raw, scaled, edge_mask = recurrent_values(spike_vector, destination, incoming)
        vectors.append((spike_vector, destination, edge_sum, scaled_raw, scaled, edge_mask))

    for destination in range(NEURONS):
        add(0, destination)
    for destination in range(NEURONS):
        add(1 << 7, destination)

    positive_destination = max(
        incoming,
        key=lambda destination: sum(sign > 0 for _, sign in incoming[destination]),
    )
    positive_sources = [source for source, sign in incoming[positive_destination] if sign > 0]
    add(sum(1 << source for source in positive_sources), positive_destination)

    negative_destination = max(
        incoming,
        key=lambda destination: sum(sign < 0 for _, sign in incoming[destination]),
    )
    negative_sources = [source for source, sign in incoming[negative_destination] if sign < 0]
    add(sum(1 << source for source in negative_sources), negative_destination)

    mixed_destination = max(
        (destination for destination in incoming if any(sign > 0 for _, sign in incoming[destination])
         and any(sign < 0 for _, sign in incoming[destination])),
        key=lambda destination: len(incoming[destination]),
    )
    add(sum(1 << source for source, _ in incoming[mixed_destination]), mixed_destination)

    maximum_destination = max(incoming, key=lambda destination: len(incoming[destination]))
    add(sum(1 << source for source, _ in incoming[maximum_destination]), maximum_destination)
    return vectors


def build_reservoir_step_vectors(incoming):
    with np.load(VECTOR_DIR / "golden_vectors.npz", allow_pickle=False) as vectors:
        threshold = vectors["threshold_result"]
        membrane_before = vectors["membrane_before"]
        input_contribution = vectors["input_contribution"]
        expected_after = vectors["membrane_after_update"]
        expected_spike = vectors["threshold_result"]
        expected_reset = vectors["reset_membrane"]
        expected_raw_from_model = vectors["recurrent_contribution_raw"]
        expected_recurrent_from_model = vectors["recurrent_contribution"]

    result = []
    for sample in range(threshold.shape[0]):
        for timestep in range(threshold.shape[1]):
            previous = 0
            if timestep:
                previous = sum(int(threshold[sample, timestep - 1, source]) << source for source in range(NEURONS))
            for destination in range(NEURONS):
                edge_sum, scaled_raw, scaled, edge_mask = recurrent_values(previous, destination, incoming)
                if int(expected_raw_from_model[sample, timestep, destination]) != scaled_raw:
                    raise RuntimeError("Derived recurrent raw value does not match the golden model")
                if int(expected_recurrent_from_model[sample, timestep, destination]) != scaled:
                    raise RuntimeError("Derived recurrent contribution does not match the golden model")
                result.append(
                    (
                        previous,
                        destination,
                        int(membrane_before[sample, timestep, destination]),
                        int(input_contribution[sample, timestep, destination]),
                        edge_sum,
                        scaled_raw,
                        scaled,
                        int(expected_after[sample, timestep, destination]),
                        int(expected_spike[sample, timestep, destination]),
                        int(expected_reset[sample, timestep, destination]),
                        edge_mask,
                    )
                )
    return result


def write_graph_memories(edges) -> None:
    write_mem(MEM_DIR / "recurrent_edge_source.mem", [source for source, _, _ in edges], 6)
    write_mem(MEM_DIR / "recurrent_edge_destination.mem", [destination for _, destination, _ in edges], 6)
    write_mem(MEM_DIR / "recurrent_edge_sign.mem", [0 if sign > 0 else 1 for _, _, sign in edges], 1)


def write_unit_memories(vectors) -> None:
    write_mem(MEM_DIR / "recurrent_unit_spikes.mem", [row[0] for row in vectors], 64)
    write_mem(MEM_DIR / "recurrent_unit_destination.mem", [row[1] for row in vectors], 6)
    write_mem(MEM_DIR / "recurrent_unit_expected_sum.mem", [row[2] for row in vectors], 5)
    write_mem(MEM_DIR / "recurrent_unit_expected_raw.mem", [row[3] for row in vectors], 32)
    write_mem(MEM_DIR / "recurrent_unit_expected_contribution.mem", [row[4] for row in vectors], 16)
    write_mem(MEM_DIR / "recurrent_unit_expected_edge_mask.mem", [row[5] for row in vectors], 64)


def write_step_memories(vectors) -> None:
    names_bits = (
        ("reservoir_step_spikes", 64, 0),
        ("reservoir_step_destination", 6, 1),
        ("reservoir_step_membrane_before", 16, 2),
        ("reservoir_step_input", 32, 3),
        ("reservoir_step_expected_sum", 5, 4),
        ("reservoir_step_expected_raw", 32, 5),
        ("reservoir_step_expected_recurrent", 32, 6),
        ("reservoir_step_expected_after", 16, 7),
        ("reservoir_step_expected_spike", 1, 8),
        ("reservoir_step_expected_reset", 16, 9),
        ("reservoir_step_expected_edge_mask", 64, 10),
    )
    for name, bits, index in names_bits:
        write_mem(MEM_DIR / f"{name}.mem", [row[index] for row in vectors], bits)


def compile_and_run(iverilog: str, vvp: str, top: str, parameter: str, sources: list[str], output_name: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="recurrent_tb_") as temp_dir:
        sim_path = Path(temp_dir) / output_name
        compile_cmd = [
            iverilog,
            "-g2012",
            "-s",
            top,
            parameter,
            "-o",
            str(sim_path),
            *sources,
        ]
        compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
        if compiled.returncode != 0:
            return compiled.returncode, compiled.stdout + compiled.stderr
        executed = subprocess.run([vvp, str(sim_path)], cwd=ROOT, text=True, capture_output=True)
        return executed.returncode, executed.stdout + executed.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recurrent RTL regression")
    parser.add_argument("--smoke", action="store_true", help="use a small deterministic reservoir-step subset")
    args = parser.parse_args()

    iverilog, vvp, simulator = find_simulator()
    if not iverilog or not vvp:
        print("SKIP: iverilog/vvp simulator not available; recurrent RTL was not executed")
        return 2

    edges, incoming = load_edges()
    unit_vectors = build_unit_vectors(incoming)
    step_vectors = build_reservoir_step_vectors(incoming)
    if args.smoke:
        step_vectors = step_vectors[:64] + step_vectors[-64:]
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    write_graph_memories(edges)
    write_unit_memories(unit_vectors)
    write_step_memories(step_vectors)

    print(f"Simulator: {simulator}; {simulator_version(iverilog)}")
    print(f"Graph edges: {len(edges)}; unit vectors: {len(unit_vectors)}; golden neuron vectors: {len(step_vectors)}")

    unit_code, unit_output = compile_and_run(
        iverilog,
        vvp,
        "tb_sparse_recurrent_engine",
        f"-Ptb_sparse_recurrent_engine.NUM_VECTORS={len(unit_vectors)}",
        [
            "rtl/src/fixed_point_pkg.sv",
            "rtl/src/sparse_recurrent_engine.sv",
            "rtl/tb/tb_sparse_recurrent_engine.sv",
        ],
        "tb_sparse_recurrent_engine.vvp",
    )
    print(unit_output, end="")
    if unit_code != 0 or "PASS:" not in unit_output:
        return unit_code or 1

    step_code, step_output = compile_and_run(
        iverilog,
        vvp,
        "tb_reservoir_step",
        f"-Ptb_reservoir_step.NUM_VECTORS={len(step_vectors)}",
        [
            "rtl/src/fixed_point_pkg.sv",
            "rtl/src/lif_pe.sv",
            "rtl/src/sparse_recurrent_engine.sv",
            "rtl/src/reservoir_step.sv",
            "rtl/tb/tb_reservoir_step.sv",
        ],
        "tb_reservoir_step.vvp",
    )
    print(step_output, end="")
    if step_code != 0 or "PASS:" not in step_output:
        return step_code or 1

    print(
        f"Recurrent comparisons: {len(unit_vectors) * 3} unit exact; "
        f"{len(step_vectors) * 3} golden recurrent exact"
    )
    print(f"Reservoir-step comparisons: {len(step_vectors) * 7} exact")
    print("Mismatches: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
