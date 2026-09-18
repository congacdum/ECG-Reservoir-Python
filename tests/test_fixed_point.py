from pathlib import Path
import numpy as np

from src.data.loader import dataframe_to_xy, load_dataframe
from src.hardware.fixed_point import FixedPointFormat, quantize_fixed
from src.hardware.hardware_model import HardwareParams, IntegerLIFReservoir
from src.model.reservoir_numpy import NumpyLIFReservoir, ReservoirParams

ROOT = Path(__file__).resolve().parents[1]


def test_input_q_format_range_and_resolution():
    fmt = FixedPointFormat(bits=12, fractional_bits=10, signed=True)
    q = quantize_fixed(np.array([-3.0, -0.5, 0.5, 3.0]), fmt)
    assert int(q.min()) >= -2048
    assert int(q.max()) <= 2047


def test_integer_features_close_to_float_reference():
    df = load_dataframe(ROOT / "data" / "splits" / "train.csv").iloc[:256]
    X, _ = dataframe_to_xy(df)
    p = ReservoirParams(seed=42)
    f_res = NumpyLIFReservoir(p)
    float_features = f_res.transform(X)
    i_res = IntegerLIFReservoir(f_res.w_in, f_res.w_res, HardwareParams())
    int_features = i_res.transform(X)
    mismatch = np.mean(float_features != int_features)
    assert mismatch < 0.02
    assert int(int_features.max()) <= 20
