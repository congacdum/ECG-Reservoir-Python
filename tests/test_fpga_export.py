from pathlib import Path
import numpy as np

from src.hardware.export_fpga import export_hardware_artifacts


def test_fpga_export_files(tmp_path: Path):
    w_in = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    w_res = np.array([[0, 1, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.int8)
    w_out = np.array([-2, 0, 127], dtype=np.int8)
    export_hardware_artifacts(tmp_path, w_in, w_res, w_out, {"bias": False})
    assert (tmp_path / "w_in_codes.mem").exists()
    assert (tmp_path / "w_res_edges.csv").exists()
    assert (tmp_path / "w_out_int8.mem").exists()
    assert (tmp_path / "fpga_export_metadata.json").exists()
