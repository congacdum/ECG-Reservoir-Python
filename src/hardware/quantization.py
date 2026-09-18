from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SymmetricQuantizedVector:
    values: np.ndarray
    scale: float
    bits: int


def quantize_symmetric(values: np.ndarray, bits: int = 8) -> SymmetricQuantizedVector:
    if bits < 2:
        raise ValueError("Signed symmetric quantization requires at least 2 bits")
    values = np.asarray(values, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs == 0:
        return SymmetricQuantizedVector(np.zeros_like(values, dtype=np.int8 if bits <= 8 else np.int16), 1.0, bits)
    scale = max_abs / qmax
    quantized = np.clip(np.rint(values / scale), -qmax, qmax)
    dtype = np.int8 if bits <= 8 else np.int16 if bits <= 16 else np.int32
    return SymmetricQuantizedVector(quantized.astype(dtype), float(scale), bits)


def dequantize(q: SymmetricQuantizedVector) -> np.ndarray:
    return q.values.astype(np.float64) * q.scale
