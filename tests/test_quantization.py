import numpy as np
from src.hardware.quantization import quantize_symmetric, dequantize


def test_int8_quantization_is_symmetric_and_bounded():
    x = np.array([-1.0, -0.25, 0.0, 0.4, 1.2])
    q = quantize_symmetric(x, bits=8)
    assert q.values.dtype == np.int8
    assert q.values.min() >= -127
    assert q.values.max() <= 127
    recon = dequantize(q)
    assert np.max(np.abs(recon - x)) < 0.01
