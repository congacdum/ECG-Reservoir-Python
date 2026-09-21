#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
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


def _previous_spike_vector(threshold, sample: int, timestep: int) -> int:
    if timestep == 0:
        return 0
    return sum(
        int(threshold[sample, timestep - 1, source]) << source
        for source in range(NEURONS)
    )


def select_smoke_cases(threshold, incoming) -> list[tuple[int, int, int]]:
    candidates = []
    for sample in range(threshold.shape[0]):
        for timestep in range(1, threshold.shape[1]):
            previous = _previous_spike_vector(threshold, sample, timestep)
            population = previous.bit_count()
            for destination in range(NEURONS):
                active = [
                    (source, sign)
                    for source, sign in incoming[destination]
                    if (previous >> source) & 1
                ]
                signs = {sign for _, sign in active}
                candidates.append(
                    (
                        sample,
                        timestep,
                        destination,
                        population,
                        sum(sign for _, sign in active),
                        len(active),
                        len(incoming[destination]),
                        int(1 in signs),
                        int(-1 in signs),
                    )
                )

    selected: list[tuple[int, int, int]] = []
    seen = set()

    def add(case: tuple[int, int, int]) -> None:
        if case not in seen:
            selected.append(case)
            seen.add(case)

    min_fanin_destination = min(incoming, key=lambda destination: len(incoming[destination]))
    max_fanin_destination = max(incoming, key=lambda destination: len(incoming[destination]))
    add((0, 0, min_fanin_destination))
    add((0, 0, max_fanin_destination))

    sparse = min(
        (case for case in candidates if case[3] > 0),
        key=lambda case: (case[3], -case[5], case[0], case[1], case[2]),
    )
    dense = max(
        candidates,
        key=lambda case: (case[3], case[5], case[6], -case[0], -case[1], -case[2]),
    )
    positive = max(
        (case for case in candidates if case[4] > 0),
        key=lambda case: (case[5], case[6], case[3], -case[0], -case[1], -case[2]),
    )
    negative = max(
        (case for case in candidates if case[4] < 0),
        key=lambda case: (case[5], case[6], case[3], -case[0], -case[1], -case[2]),
    )
    mixed = max(
        (case for case in candidates if case[7] and case[8]),
        key=lambda case: (case[5], case[6], case[3], -case[0], -case[1], -case[2]),
    )
    add(sparse[:3])
    add(dense[:3])
    add(positive[:3])
    add(negative[:3])
    add(mixed[:3])

    for destination in (min_fanin_destination, max_fanin_destination):
        destination_cases = [case for case in candidates if case[2] == destination]
        add(max(destination_cases, key=lambda case: (case[3], case[5], -case[0], -case[1]))[:3])

    for case in candidates:
        add(case[:3])
        if len(selected) >= 8:
            break

    if len(selected) < 8 or len(selected) > 32:
        raise RuntimeError(f"Unexpected recurrent smoke case count: {len(selected)}")
    return selected


def build_reservoir_step_vectors(incoming, *, smoke: bool = False):
    with np.load(VECTOR_DIR / "golden_vectors.npz", allow_pickle=False) as vectors:
        threshold = vectors["threshold_result"]
        membrane_before = vectors["membrane_before"]
        input_contribution = vectors["input_contribution"]
        expected_after = vectors["membrane_after_update"]
        expected_spike = vectors["threshold_result"]
        expected_reset = vectors["reset_membrane"]
        expected_raw_from_model = vectors["recurrent_contribution_raw"]
        expected_recurrent_from_model = vectors["recurrent_contribution"]

    if smoke:
        cases = select_smoke_cases(threshold, incoming)
    else:
        cases = (
            (sample, timestep, destination)
            for sample in range(threshold.shape[0])
            for timestep in range(threshold.shape[1])
            for destination in range(NEURONS)
        )

    previous_cache = {}
    result = []
    for sample, timestep, destination in cases:
        case_key = (sample, timestep)
        if case_key not in previous_cache:
            previous_cache[case_key] = _previous_spike_vector(threshold, sample, timestep)
        previous = previous_cache[case_key]
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


def compile_and_run(
    iverilog: str,
    vvp: str,
    top: str,
    parameter: str,
    sources: list[str],
    output_name: str,
    phase: str,
) -> tuple[int, str, float, float]:
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
        print(f"[recurrent] {phase} compile start", flush=True)
        compile_started = time.perf_counter()
        compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
        compile_seconds = time.perf_counter() - compile_started
        print(f"[recurrent] {phase} compile finished in {compile_seconds:.3f}s", flush=True)
        if compiled.returncode != 0:
            return compiled.returncode, compiled.stdout + compiled.stderr, compile_seconds, 0.0
        print(f"[recurrent] {phase} simulation start", flush=True)
        simulation_started = time.perf_counter()
        executed = subprocess.run([vvp, str(sim_path)], cwd=ROOT, text=True, capture_output=True)
        simulation_seconds = time.perf_counter() - simulation_started
        print(f"[recurrent] {phase} simulation finished in {simulation_seconds:.3f}s", flush=True)
        return executed.returncode, executed.stdout + executed.stderr, compile_seconds, simulation_seconds


def main() -> int:
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run the recurrent RTL regression")
    parser.add_argument("--smoke", action="store_true", help="use a small deterministic reservoir-step subset")
    args = parser.parse_args()

    iverilog, vvp, simulator = find_simulator()
    if not iverilog or not vvp:
        print("SKIP: iverilog/vvp simulator not available; recurrent RTL was not executed")
        return 2

    mode = "smoke" if args.smoke else "full"
    print(f"[recurrent] mode={mode}", flush=True)
    generation_started = time.perf_counter()
    edges, incoming = load_edges()
    unit_vectors = build_unit_vectors(incoming)
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    write_graph_memories(edges)
    write_unit_memories(unit_vectors)
    unit_generation_seconds = time.perf_counter() - generation_started

    print(f"Simulator: {simulator}; {simulator_version(iverilog)}")
    print(f"Graph edges: {len(edges)}; unit vectors: {len(unit_vectors)}")
    print(f"[recurrent] unit generation finished in {unit_generation_seconds:.3f}s", flush=True)

    print("[recurrent] golden generation start", flush=True)
    golden_generation_started = time.perf_counter()
    step_vectors = build_reservoir_step_vectors(incoming, smoke=args.smoke)
    write_step_memories(step_vectors)
    golden_generation_seconds = time.perf_counter() - golden_generation_started
    print(f"[recurrent] golden vectors={len(step_vectors)}", flush=True)
    if args.smoke:
        print(f"[recurrent] smoke cases={len(step_vectors)}", flush=True)
    print(f"[recurrent] vectors={len(step_vectors)}", flush=True)
    print(f"[recurrent] expected comparisons={len(step_vectors) * 7}", flush=True)
    print(f"[recurrent] golden generation finished in {golden_generation_seconds:.3f}s", flush=True)

    unit_code, unit_output, unit_compile_seconds, unit_simulation_seconds = compile_and_run(
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
        "unit",
    )
    unit_compare_started = time.perf_counter()
    print(unit_output, end="")
    if unit_code != 0 or "PASS:" not in unit_output:
        return unit_code or 1
    print(f"[recurrent] unit comparison finished in {time.perf_counter() - unit_compare_started:.3f}s", flush=True)
    print("[recurrent] golden phase starting", flush=True)
    print(f"[recurrent] golden vectors={len(step_vectors)}", flush=True)

    step_code, step_output, step_compile_seconds, step_simulation_seconds = compile_and_run(
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
        "golden",
    )
    print("[recurrent] golden parsing start", flush=True)
    comparison_started = time.perf_counter()
    print(step_output, end="")
    parse_seconds = time.perf_counter() - comparison_started
    print(f"[recurrent] golden parsing finished in {parse_seconds:.3f}s", flush=True)
    golden_compare_started = time.perf_counter()
    golden_ok = step_code == 0 and "PASS:" in step_output
    print(f"[recurrent] golden comparison finished in {time.perf_counter() - golden_compare_started:.3f}s", flush=True)
    if not golden_ok:
        return step_code or 1

    print(
        f"Recurrent comparisons: {len(unit_vectors) * 3} unit exact; "
        f"{len(step_vectors) * 3} golden recurrent exact"
    )
    print(f"Reservoir-step comparisons: {len(step_vectors) * 7} exact")
    print("Mismatches: 0")
    print(f"[recurrent] total runtime {time.perf_counter() - total_started:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
