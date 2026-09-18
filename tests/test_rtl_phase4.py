import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL_SRC = ROOT / "rtl" / "src"
WEIGHTS = ROOT / "outputs" / "fpga" / "weights" / "w_out_int8.mem"
METADATA = ROOT / "outputs" / "fpga" / "weights" / "fpga_export_metadata.json"
MANIFEST = ROOT / "outputs" / "fpga" / "golden_vectors" / "golden_vectors_manifest.json"


def _simulator_available() -> bool:
    if shutil.which("iverilog") and shutil.which("vvp"):
        return True
    return (
        Path(r"C:\msys64\ucrt64\bin\iverilog.exe").exists()
        and Path(r"C:\msys64\ucrt64\bin\vvp.exe").exists()
    )


def _run(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "rtl" / "scripts" / script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def test_readout_architecture_and_width_proof():
    readout = (RTL_SRC / "readout_mac.sv").read_text(encoding="utf-8")
    core = (RTL_SRC / "ecg_classifier_core.sv").read_text(encoding="utf-8")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    weights = [int(token, 16) for token in WEIGHTS.read_text(encoding="ascii").split()]

    assert "NEURONS = 64" in readout
    assert "COUNT_BITS = 5" in readout
    assert "ACCUMULATOR_BITS = 20" in readout
    assert "w_out_int8.mem" in readout
    assert "predicted_class <= (accumulator_next >= 0)" in readout
    assert "readout_start <= reservoir_done" in core
    assert "bias" not in readout.lower()
    assert "bias" not in core.lower()
    assert len(weights) == 64
    signed_weights = [value - 256 if value & 0x80 else value for value in weights]
    assert all(-128 <= value <= 127 for value in signed_weights)
    assert metadata["readout_weight_bits"] == 8
    assert metadata["accumulator_bits"] == 20
    assert metadata["bias"] is False
    assert manifest["fixed_point"]["readout_accumulator"]["bits"] == 20
    assert manifest["no_bias"]["readout_intercept"] is False

    # Counts are 0..20 and weights are signed INT8. The minimum signed width
    # that contains the theoretical bounds is 19; the locked implementation
    # uses 20 bits, leaving one additional safety bit.
    lower = 64 * 20 * -128
    upper = 64 * 20 * 127
    assert lower == -163840
    assert upper == 162560
    assert -(1 << 17) > lower
    assert (1 << 17) - 1 < upper
    assert -(1 << 19) <= lower <= upper <= (1 << 19) - 1
    assert -(1 << 20) <= lower <= upper <= (1 << 20) - 1


@pytest.mark.skipif(not _simulator_available(), reason="iverilog/vvp simulator not available")
def test_readout_mac_regression():
    output = _run("run_readout_tb.py")
    assert "PASS: readout vectors=14 MAC comparisons=896" in output
    assert "Zero-score rule: class 1 verified" in output
    assert "Mismatches: 0" in output


@pytest.mark.skipif(not _simulator_available(), reason="iverilog/vvp simulator not available")
def test_full_classifier_regression():
    output = _run("run_classifier_tb.py")
    assert "PASS: full classifier samples=8" in output
    assert "cycles_from_start_to_done=1386" in output
    assert "Mismatches: 0" in output
