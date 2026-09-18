import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL_SRC = ROOT / "rtl" / "src"
MANIFEST = ROOT / "outputs" / "fpga" / "golden_vectors" / "golden_vectors_manifest.json"


def test_rtl_phase1_files_exist():
    assert (RTL_SRC / "fixed_point_pkg.sv").exists()
    assert (RTL_SRC / "lif_pe.sv").exists()
    assert (ROOT / "rtl" / "tb" / "tb_lif_pe.sv").exists()
    assert (ROOT / "rtl" / "scripts" / "run_lif_tb.py").exists()


def test_rtl_manifest_widths_and_no_bias_terms():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["fixed_point"]["input"]["bits"] == 12
    assert manifest["fixed_point"]["membrane"]["bits"] == 16
    assert manifest["fixed_point"]["spike_count"]["bits"] == 5
    assert manifest["fixed_point"]["readout_accumulator"]["bits"] == 20
    assert manifest["no_bias"] == {
        "reservoir_bias_current": False,
        "readout_intercept": False,
        "exported_bias": False,
    }
    rtl_text = "\n".join(path.read_text(encoding="utf-8").lower() for path in RTL_SRC.glob("*.sv"))
    assert "bias" not in rtl_text
    assert "i_bias" not in rtl_text
    lif_text = (RTL_SRC / "lif_pe.sv").read_text(encoding="utf-8")
    assert "logic signed [15:0] membrane_in" in lif_text
    assert "logic signed [31:0] input_contribution" in lif_text
    assert "logic signed [31:0] recurrent_contribution" in lif_text
    assert "THRESHOLD_Q = 16'sd2560" in lif_text
    assert "LEAK_SHIFT = 3" in lif_text


def test_rtl_lif_regression_if_simulator_available():
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        pytest.skip("iverilog/vvp simulator not available")
    result = subprocess.run(
        [sys.executable, str(ROOT / "rtl" / "scripts" / "run_lif_tb.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
