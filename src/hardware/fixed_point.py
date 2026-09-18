from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FixedPointFormat:
    bits: int
    fractional_bits: int
    signed: bool = True

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def min_int(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max_int(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1


def quantize_fixed(values, fmt: FixedPointFormat) -> np.ndarray:
    q = np.rint(np.asarray(values, dtype=np.float64) * fmt.scale)
    q = np.clip(q, fmt.min_int, fmt.max_int)
    dtype = np.int16 if fmt.bits <= 16 else np.int32 if fmt.bits <= 32 else np.int64
    return q.astype(dtype)


def quantization_diagnostics(values, fmt: FixedPointFormat) -> dict:
    """Return measured clipping and non-zero-to-zero quantization statistics."""
    values = np.asarray(values, dtype=np.float64)
    scaled = values * fmt.scale
    rounded = np.rint(scaled)
    clipped = (scaled < fmt.min_int) | (scaled > fmt.max_int)
    nonzero = values != 0
    underflow = nonzero & (rounded == 0)
    nonzero_count = int(np.count_nonzero(nonzero))
    total = int(values.size)
    return {
        "input_clipping_count": int(np.count_nonzero(clipped)),
        "input_clipping_rate": float(np.count_nonzero(clipped) / total) if total else 0.0,
        "input_underflow_to_zero_count": int(np.count_nonzero(underflow)),
        "input_underflow_to_zero_rate": float(np.count_nonzero(underflow) / nonzero_count)
        if nonzero_count
        else 0.0,
        "input_value_count": total,
        "input_nonzero_value_count": nonzero_count,
    }


def dequantize_fixed(values, fmt: FixedPointFormat) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / fmt.scale


def saturate(values, bits: int, signed: bool = True) -> np.ndarray:
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    return np.clip(values, lo, hi)
