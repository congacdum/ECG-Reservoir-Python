from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fixed_point import FixedPointFormat, quantization_diagnostics, quantize_fixed, saturate


@dataclass(frozen=True)
class HardwareParams:
    input_format: FixedPointFormat = FixedPointFormat(bits=12, fractional_bits=10, signed=True)
    membrane_bits: int = 16
    membrane_fractional_bits: int = 10
    threshold: float = 2.5
    leak_shift: int = 3          # 7/8: v - (v >> 3)
    recurrent_shift: int = 4     # 1/16
    recurrent_accumulator_bits: int = 16
    saturation: bool = True


class IntegerLIFReservoir:
    """Bit-oriented golden model for the FPGA reservoir.

    Exact choices used here:
    - signed 12-bit Q1.10 input
    - signed 16-bit membrane with 10 fractional bits
    - input gain 3/2 combined with Win in {1/2,1,2}
      * Win=1/2 => input * 3/4
      * Win=1   => input * 3/2
      * Win=2   => input * 3
    - leak 7/8 by arithmetic right shift
    - recurrent ternary sum scaled by 1/16
    - threshold/reset LIF
    - 5-bit-compatible spike counts for 20 timesteps
    """

    def __init__(self, w_in: np.ndarray, w_res: np.ndarray, params: HardwareParams | None = None):
        self.w_in = np.asarray(w_in, dtype=np.float32)
        self.w_res = np.asarray(w_res, dtype=np.int8)
        self.params = params or HardwareParams()
        n = self.w_in.size
        if self.w_res.shape != (n, n):
            raise ValueError("w_res shape mismatch")
        if not np.all(np.isin(self.w_in, (0.5, 1.0, 2.0))):
            raise ValueError("Hardware Win must be in {0.5,1,2}")
        if not np.all(np.isin(self.w_res, (-1, 0, 1))):
            raise ValueError("Hardware Wres must be ternary")
        self._input_codes = np.where(self.w_in == 0.5, 0, np.where(self.w_in == 1.0, 1, 2)).astype(np.int8)

    @property
    def neurons(self) -> int:
        return int(self.w_in.size)

    def _input_current(self, xq: np.ndarray) -> np.ndarray:
        # xq shape [batch, 1]. x3 = 3*x uses shift+add only.
        x3 = xq + (xq << 1)
        out = np.empty((xq.shape[0], self.neurons), dtype=np.int32)
        for j, code in enumerate(self._input_codes):
            if code == 0:
                out[:, j] = x3[:, 0] >> 2  # 3/4
            elif code == 1:
                out[:, j] = x3[:, 0] >> 1  # 3/2
            else:
                out[:, j] = x3[:, 0]       # 3
        return out

    def transform(
        self,
        X: np.ndarray,
        return_spikes: bool = False,
        return_diagnostics: bool = False,
    ):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must have shape [batch,time]")
        p = self.params
        xq = quantize_fixed(X, p.input_format).astype(np.int32)
        input_stats = quantization_diagnostics(X, p.input_format)
        scale = 1 << p.membrane_fractional_bits
        threshold_q = int(round(p.threshold * scale))
        batch, timesteps = X.shape
        v = np.zeros((batch, self.neurons), dtype=np.int32)
        spikes = np.zeros_like(v)
        counts = np.zeros((batch, self.neurons), dtype=np.uint8)
        history = np.zeros((batch, timesteps, self.neurons), dtype=np.uint8) if return_spikes else None
        membrane_upper = 0
        membrane_lower = 0
        recurrent_upper = 0
        recurrent_lower = 0
        membrane_updates = batch * timesteps * self.neurons
        recurrent_updates = membrane_updates
        wres32 = self.w_res.T.astype(np.int32)
        rec_unit = scale >> p.recurrent_shift
        membrane_min = -(1 << (p.membrane_bits - 1))
        membrane_max = (1 << (p.membrane_bits - 1)) - 1
        recurrent_min = -(1 << (p.recurrent_accumulator_bits - 1))
        recurrent_max = (1 << (p.recurrent_accumulator_bits - 1)) - 1

        for t in range(timesteps):
            inp = self._input_current(xq[:, t, None])
            recurrent_sum = spikes @ wres32
            recurrent_raw = recurrent_sum * rec_unit
            recurrent_upper += int(np.count_nonzero(recurrent_raw > recurrent_max))
            recurrent_lower += int(np.count_nonzero(recurrent_raw < recurrent_min))
            recurrent = (
                saturate(recurrent_raw, p.recurrent_accumulator_bits, signed=True).astype(np.int32)
                if p.saturation
                else recurrent_raw
            )
            # arithmetic right shift on signed int32 matches intended two's-complement style
            v_raw = (v - (v >> p.leak_shift)) + inp + recurrent
            membrane_upper += int(np.count_nonzero(v_raw > membrane_max))
            membrane_lower += int(np.count_nonzero(v_raw < membrane_min))
            if p.saturation:
                v = saturate(v_raw, p.membrane_bits, signed=True).astype(np.int32)
            else:
                v = v_raw
            spikes = (v >= threshold_q).astype(np.int32)
            counts += spikes.astype(np.uint8)
            if history is not None:
                history[:, t, :] = spikes.astype(np.uint8)
            v = np.where(spikes > 0, 0, v)

        diagnostics = {
            **input_stats,
            "membrane_upper_saturation_count": membrane_upper,
            "membrane_upper_saturation_rate": float(membrane_upper / membrane_updates)
            if membrane_updates
            else 0.0,
            "membrane_lower_saturation_count": membrane_lower,
            "membrane_lower_saturation_rate": float(membrane_lower / membrane_updates)
            if membrane_updates
            else 0.0,
            "recurrent_accumulator_upper_saturation_count": recurrent_upper,
            "recurrent_accumulator_upper_saturation_rate": float(recurrent_upper / recurrent_updates)
            if recurrent_updates
            else 0.0,
            "recurrent_accumulator_lower_saturation_count": recurrent_lower,
            "recurrent_accumulator_lower_saturation_rate": float(recurrent_lower / recurrent_updates)
            if recurrent_updates
            else 0.0,
            "membrane_update_count": membrane_updates,
            "recurrent_accumulator_update_count": recurrent_updates,
            "saturation_enabled": bool(p.saturation),
        }
        if return_spikes and return_diagnostics:
            return counts, history, diagnostics
        if return_spikes:
            return counts, history
        if return_diagnostics:
            return counts, diagnostics
        return counts

    def transform_trace(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return bit-exact intermediate states for RTL golden-vector export."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must have shape [batch,time]")
        p = self.params
        xq = quantize_fixed(X, p.input_format).astype(np.int32)
        scale = 1 << p.membrane_fractional_bits
        threshold_q = int(round(p.threshold * scale))
        batch, timesteps = X.shape
        v = np.zeros((batch, self.neurons), dtype=np.int32)
        spikes = np.zeros_like(v)
        counts = np.zeros((batch, self.neurons), dtype=np.uint8)
        shape = (batch, timesteps, self.neurons)
        membrane_before = np.zeros(shape, dtype=np.int32)
        input_contribution = np.zeros(shape, dtype=np.int32)
        recurrent_contribution = np.zeros(shape, dtype=np.int32)
        recurrent_contribution_raw = np.zeros(shape, dtype=np.int32)
        membrane_update_raw = np.zeros(shape, dtype=np.int32)
        membrane_after_update = np.zeros(shape, dtype=np.int32)
        threshold_result = np.zeros(shape, dtype=np.uint8)
        reset_membrane = np.zeros(shape, dtype=np.int32)
        spike_count_state = np.zeros(shape, dtype=np.uint8)
        wres32 = self.w_res.T.astype(np.int32)
        rec_unit = scale >> p.recurrent_shift
        membrane_min = -(1 << (p.membrane_bits - 1))
        membrane_max = (1 << (p.membrane_bits - 1)) - 1

        for t in range(timesteps):
            inp = self._input_current(xq[:, t, None])
            recurrent_sum = spikes @ wres32
            recurrent_raw = recurrent_sum * rec_unit
            recurrent = (
                saturate(recurrent_raw, p.recurrent_accumulator_bits, signed=True).astype(np.int32)
                if p.saturation
                else recurrent_raw
            )
            membrane_update = (v - (v >> p.leak_shift)) + inp + recurrent
            if p.saturation:
                v_after = saturate(membrane_update, p.membrane_bits, signed=True).astype(np.int32)
            else:
                v_after = membrane_update
            current_spikes = (v_after >= threshold_q).astype(np.uint8)
            counts += current_spikes
            v_reset = np.where(current_spikes > 0, 0, v_after).astype(np.int32)

            membrane_before[:, t, :] = v
            input_contribution[:, t, :] = inp
            recurrent_contribution[:, t, :] = recurrent
            recurrent_contribution_raw[:, t, :] = recurrent_raw
            membrane_update_raw[:, t, :] = membrane_update
            membrane_after_update[:, t, :] = v_after
            threshold_result[:, t, :] = current_spikes
            reset_membrane[:, t, :] = v_reset
            spike_count_state[:, t, :] = counts
            v = v_reset
            spikes = current_spikes.astype(np.int32)

        return {
            "input_quantized": xq,
            "membrane_before": membrane_before,
            "input_contribution": input_contribution,
            "recurrent_contribution": recurrent_contribution,
            "recurrent_contribution_raw": recurrent_contribution_raw,
            "membrane_update_raw": membrane_update_raw,
            "membrane_after_update": membrane_after_update,
            "threshold_result": threshold_result,
            "reset_membrane": reset_membrane,
            "spike_count_state": spike_count_state,
            "spike_counts": counts,
            "membrane_min": np.asarray([membrane_min], dtype=np.int32),
            "membrane_max": np.asarray([membrane_max], dtype=np.int32),
            "threshold_q": np.asarray([threshold_q], dtype=np.int32),
        }


def integer_readout_score(features: np.ndarray, w_out_int: np.ndarray, accumulator_bits: int = 20) -> np.ndarray:
    features = np.asarray(features, dtype=np.int32)
    weights = np.asarray(w_out_int, dtype=np.int32).reshape(-1)
    if features.shape[1] != weights.size:
        raise ValueError("Feature/weight dimension mismatch")
    score = features @ weights
    return saturate(score, accumulator_bits, signed=True).astype(np.int32)


def integer_readout_diagnostics(
    features: np.ndarray,
    w_out_int: np.ndarray,
    accumulator_bits: int = 20,
) -> dict:
    """Measure clipping that would occur in the signed readout accumulator."""
    features = np.asarray(features, dtype=np.int32)
    weights = np.asarray(w_out_int, dtype=np.int32).reshape(-1)
    if features.ndim != 2 or features.shape[1] != weights.size:
        raise ValueError("Feature/weight dimension mismatch")
    raw_score = features @ weights
    lo = -(1 << (accumulator_bits - 1))
    hi = (1 << (accumulator_bits - 1)) - 1
    upper = int(np.count_nonzero(raw_score > hi))
    lower = int(np.count_nonzero(raw_score < lo))
    total = int(raw_score.size)
    return {
        "readout_accumulator_upper_saturation_count": upper,
        "readout_accumulator_upper_saturation_rate": float(upper / total) if total else 0.0,
        "readout_accumulator_lower_saturation_count": lower,
        "readout_accumulator_lower_saturation_rate": float(lower / total) if total else 0.0,
        "readout_accumulator_sample_count": total,
        "readout_accumulator_bits": int(accumulator_bits),
    }


def integer_predict(features: np.ndarray, w_out_int: np.ndarray, accumulator_bits: int = 20) -> tuple[np.ndarray, np.ndarray]:
    score = integer_readout_score(features, w_out_int, accumulator_bits)
    pred = (score >= 0).astype(np.int64)
    return pred, score
