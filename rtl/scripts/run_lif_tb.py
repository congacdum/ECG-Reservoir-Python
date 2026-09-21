#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEM_DIR = ROOT / "rtl" / "mem"
VECTOR_DIR = ROOT / "outputs" / "fpga" / "golden_vectors"


def find_simulator() -> tuple[str | None, str | None, str]:
    """Find a usable Icarus pair, including common native MSYS2 installs."""
    path_iverilog = shutil.which("iverilog")
    path_vvp = shutil.which("vvp")
    if path_iverilog and path_vvp:
        return path_iverilog, path_vvp, "Icarus Verilog (PATH)"

    candidates = (
        Path(r"C:\msys64\ucrt64\bin"),
        Path(r"C:\msys64\mingw64\bin"),
        Path(r"C:\msys64\clang64\bin"),
        Path(r"C:\msys2\ucrt64\bin"),
        Path(r"C:\msys2\mingw64\bin"),
        Path(r"C:\msys2\clang64\bin"),
    )
    for bin_dir in candidates:
        iverilog = bin_dir / "iverilog.exe"
        vvp = bin_dir / "vvp.exe"
        if iverilog.is_file() and vvp.is_file():
            return str(iverilog), str(vvp), f"Icarus Verilog ({bin_dir})"
    return None, None, ""


def simulator_version(iverilog: str) -> str:
    result = subprocess.run(
        [iverilog, "-V"],
        text=True,
        capture_output=True,
        check=False,
    )
    combined = (result.stdout + "\n" + result.stderr).splitlines()
    for line in combined:
        if "Icarus Verilog version" in line:
            return line.strip()
    return "version unavailable"


def unsigned_hex(value: int, bits: int) -> str:
    return f"{int(value) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def write_mem(path: Path, values, bits: int) -> None:
    with path.open("w", encoding="ascii", newline="\n") as f:
        for value in values:
            f.write(unsigned_hex(int(value), bits) + "\n")


def directed_vectors() -> list[tuple[int, int, int, int, int, int]]:
    cases = [
        (0, 0, 0),
        (0, 2560, 0),
        (0, 0, 64),
        (64, 0, -128),
        (0, 100000, 0),
        (0, -100000, 0),
    ]
    result = []
    for before, input_value, recurrent_value in cases:
        raw = (before - (before >> 3)) + input_value + recurrent_value
        after = min(32767, max(-32768, raw))
        spike = int(after >= 2560)
        reset = 0 if spike else after
        result.append((before, input_value, recurrent_value, after, spike, reset))
    return result


def prepare_memories(*, smoke: bool = False) -> tuple[int, int]:
    import numpy as np

    with np.load(VECTOR_DIR / "golden_vectors.npz", allow_pickle=False) as vectors:
        required = (
            "membrane_before",
            "input_contribution",
            "recurrent_contribution",
            "membrane_after_update",
            "threshold_result",
            "reset_membrane",
        )
        for name in required:
            if name not in vectors.files:
                raise RuntimeError(f"Missing golden-vector array: {name}")
        expected_before = vectors["membrane_before"].reshape(-1).astype(np.int64).tolist()
        input_values = vectors["input_contribution"].reshape(-1).astype(np.int64).tolist()
        recurrent_values = vectors["recurrent_contribution"].reshape(-1).astype(np.int64).tolist()
        expected_after = vectors["membrane_after_update"].reshape(-1).astype(np.int64).tolist()
        expected_spike = vectors["threshold_result"].reshape(-1).astype(np.int64).tolist()
        expected_reset = vectors["reset_membrane"].reshape(-1).astype(np.int64).tolist()

    directed = directed_vectors()
    for before, input_value, recurrent_value, after, spike, reset in directed:
        expected_before.append(before)
        input_values.append(input_value)
        recurrent_values.append(recurrent_value)
        expected_after.append(after)
        expected_spike.append(spike)
        expected_reset.append(reset)

    if smoke:
        keep_golden = min(64, len(expected_before))
        expected_before = expected_before[:keep_golden]
        input_values = input_values[:keep_golden]
        recurrent_values = recurrent_values[:keep_golden]
        expected_after = expected_after[:keep_golden]
        expected_spike = expected_spike[:keep_golden]
        expected_reset = expected_reset[:keep_golden]

    MEM_DIR.mkdir(parents=True, exist_ok=True)
    write_mem(MEM_DIR / "lif_pe_expected_before.mem", expected_before, 16)
    write_mem(MEM_DIR / "lif_pe_input_contribution.mem", input_values, 32)
    write_mem(MEM_DIR / "lif_pe_recurrent_contribution.mem", recurrent_values, 32)
    write_mem(MEM_DIR / "lif_pe_expected_after.mem", expected_after, 16)
    write_mem(MEM_DIR / "lif_pe_expected_spike.mem", expected_spike, 1)
    write_mem(MEM_DIR / "lif_pe_expected_reset.mem", expected_reset, 16)
    return len(expected_before) - len(directed), len(directed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LIF RTL regression")
    parser.add_argument("--smoke", action="store_true", help="use a small deterministic vector subset")
    args = parser.parse_args()

    iverilog, vvp, simulator = find_simulator()
    if not iverilog or not vvp:
        print("SKIP: iverilog/vvp simulator not available; RTL was not executed")
        return 2

    print(f"Simulator: {simulator}; {simulator_version(iverilog)}")
    num_golden, num_directed = prepare_memories(smoke=args.smoke)
    with tempfile.TemporaryDirectory(prefix="lif_pe_tb_") as temp_dir:
        sim_path = Path(temp_dir) / "tb_lif_pe.vvp"
        compile_cmd = [
            iverilog,
            "-g2012",
            "-s",
            "tb_lif_pe",
            f"-Ptb_lif_pe.NUM_VECTORS={num_golden}",
            f"-Ptb_lif_pe.NUM_DIRECTED={num_directed}",
            "-o",
            str(sim_path),
            "rtl/src/fixed_point_pkg.sv",
            "rtl/src/lif_pe.sv",
            "rtl/tb/tb_lif_pe.sv",
        ]
        compiled = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
        if compiled.returncode != 0:
            print(compiled.stdout)
            print(compiled.stderr, file=sys.stderr)
            return compiled.returncode
        executed = subprocess.run([vvp, str(sim_path)], cwd=ROOT, text=True, capture_output=True)
        print(executed.stdout, end="")
        if executed.stderr:
            print(executed.stderr, file=sys.stderr, end="")
        if executed.returncode != 0:
            return executed.returncode
        return 0 if "PASS:" in executed.stdout else 1


if __name__ == "__main__":
    raise SystemExit(main())
