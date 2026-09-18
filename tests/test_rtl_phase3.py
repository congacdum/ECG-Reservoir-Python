import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL_SRC = ROOT / "rtl" / "src"
MANIFEST = ROOT / "outputs" / "fpga" / "golden_vectors" / "golden_vectors_manifest.json"
INPUT_CODES = ROOT / "outputs" / "fpga" / "weights" / "w_in_codes.mem"
MODEL_FILE = ROOT / "outputs" / "runs" / "run_001" / "hardware_model.npz"


def _simulator_available() -> bool:
    if shutil.which("iverilog") and shutil.which("vvp"):
        return True
    return (
        Path(r"C:\msys64\ucrt64\bin\iverilog.exe").exists()
        and Path(r"C:\msys64\ucrt64\bin\vvp.exe").exists()
    )


def test_controller_files_and_locked_dimensions():
    controller = (RTL_SRC / "reservoir_controller.sv").read_text(encoding="utf-8")
    assert (ROOT / "rtl" / "tb" / "tb_reservoir_controller.sv").exists()
    assert (ROOT / "rtl" / "scripts" / "run_controller_tb.py").exists()
    assert "parameter integer NEURONS = 64" in controller
    assert "parameter integer TIMESTEPS = 20" in controller
    assert "logic signed [15:0] membrane_current" in controller
    assert "logic signed [15:0] membrane_next" in controller
    assert "logic [63:0] previous_spikes" in controller
    assert "logic [63:0] next_spikes" in controller
    assert "logic [4:0] spike_counts_current" in controller
    assert "logic [4:0] spike_counts_next" in controller
    assert "readout" not in controller.lower()
    assert "bias" not in controller.lower()


def test_controller_manifest_and_input_weight_mapping():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sequence_length"] == 20
    assert manifest["fixed_point"]["membrane"]["bits"] == 16
    assert manifest["fixed_point"]["spike_count"]["bits"] == 5
    assert manifest["fixed_point"]["spike_count"]["range"] == [0, 20]

    codes = [int(line.strip(), 2) for line in INPUT_CODES.read_text(encoding="ascii").splitlines()]
    assert len(codes) == 64
    assert set(codes) <= {0, 1, 2}
    with np.load(MODEL_FILE, allow_pickle=False) as model:
        weights = np.asarray(model["w_in"], dtype=np.float32)
    expected = [0 if value == 0.5 else 1 if value == 1.0 else 2 for value in weights]
    assert codes == expected


def test_controller_regression_if_simulator_available():
    if not _simulator_available():
        pytest.skip("iverilog/vvp simulator not available")
    result = subprocess.run(
        [sys.executable, str(ROOT / "rtl" / "scripts" / "run_controller_tb.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Segments: 14; checkpoints: 280" in output
    assert "0=train_0000" in output
    assert "1=validation_0000" in output
    assert "2=train_0000" in output
    assert "PASS: controller checkpoints=280 segments=14" in output
    assert "Mismatches: 0" in output
