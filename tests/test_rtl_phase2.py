import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EDGE_FILE = ROOT / "outputs" / "fpga" / "weights" / "w_res_edges.csv"
MODEL_FILE = ROOT / "outputs" / "runs" / "run_001" / "hardware_model.npz"
METADATA_FILE = ROOT / "outputs" / "fpga" / "weights" / "fpga_export_metadata.json"
RTL_SRC = ROOT / "rtl" / "src"


def _load_edges():
    with EDGE_FILE.open(newline="", encoding="ascii") as f:
        return list(csv.DictReader(f))


def _simulator_available() -> bool:
    if shutil.which("iverilog") and shutil.which("vvp"):
        return True
    return (
        Path(r"C:\msys64\ucrt64\bin\iverilog.exe").exists()
        and Path(r"C:\msys64\ucrt64\bin\vvp.exe").exists()
    )


def test_rtl_phase2_files_and_metadata_exist():
    assert (RTL_SRC / "sparse_recurrent_engine.sv").exists()
    assert (RTL_SRC / "reservoir_step.sv").exists()
    assert (ROOT / "rtl" / "tb" / "tb_sparse_recurrent_engine.sv").exists()
    assert (ROOT / "rtl" / "tb" / "tb_reservoir_step.sv").exists()
    assert (ROOT / "rtl" / "scripts" / "run_recurrent_tb.py").exists()
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    assert metadata["recurrent_edges"] == 403
    assert metadata["w_res_encoding"] == "sparse edge list; sign 0=+1, 1=-1"


def test_frozen_sparse_graph_checksum_orientation_and_statistics():
    rows = _load_edges()
    assert len(rows) == 403
    assert hashlib.sha256(EDGE_FILE.read_bytes()).hexdigest() == (
        "77c8a1529768eb86bfd1b67a744034f5613d4433cb5f7d54060614440df6449f"
    )

    reconstructed = np.zeros((64, 64), dtype=np.int8)
    pairs = set()
    fan_in = np.zeros(64, dtype=np.int32)
    positive = 0
    negative = 0
    for row in rows:
        source = int(row["source"])
        destination = int(row["destination"])
        sign_code = int(row["sign"])
        assert 0 <= source < 64
        assert 0 <= destination < 64
        assert source != destination
        assert sign_code in (0, 1)
        assert (source, destination) not in pairs
        pairs.add((source, destination))
        sign = 1 if sign_code == 0 else -1
        reconstructed[destination, source] = sign
        fan_in[destination] += 1
        positive += sign > 0
        negative += sign < 0

    with np.load(MODEL_FILE, allow_pickle=False) as model:
        w_res = np.asarray(model["w_res"], dtype=np.int8)
    assert np.array_equal(reconstructed, w_res)
    assert positive == 311
    assert negative == 92
    assert int(fan_in.max()) == 12
    assert int(fan_in.min()) == 2
    assert float(fan_in.mean()) == 6.296875


def test_recurrent_rtl_regression_if_simulator_available():
    if not _simulator_available():
        pytest.skip("iverilog/vvp simulator not available")
    result = subprocess.run(
        [sys.executable, str(ROOT / "rtl" / "scripts" / "run_recurrent_tb.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Mismatches: 0" in result.stdout
    assert "PASS: 71680 reservoir-step comparisons exact" in result.stdout
