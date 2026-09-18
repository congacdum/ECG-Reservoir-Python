from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReservoirParams:
    neurons: int = 64
    connectivity: float = 0.10
    inhibitory_fraction: float = 0.20
    input_weight_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    input_weight_probabilities: tuple[float, ...] = (0.25, 0.50, 0.25)
    leak: float = 7 / 8
    input_gain: float = 3 / 2
    recurrent_gain: float = 1 / 16
    threshold: float = 2.5
    reset_value: float = 0.0
    seed: int = 42


class NumpyLIFReservoir:
    """Deterministic, no-bias, hardware-oriented LIF reservoir.

    The reservoir is intentionally simple:
    - fixed positive power-of-two input weights {0.5, 1, 2}
    - sparse ternary recurrent weights {-1, 0, +1}
    - no trainable reservoir parameters
    - no additive bias current
    - sample state is reset between independent ECG windows
    - output feature is spike count per neuron
    """

    def __init__(self, params: ReservoirParams, w_in: np.ndarray | None = None, w_res: np.ndarray | None = None):
        self.params = params
        if params.neurons <= 0:
            raise ValueError("neurons must be > 0")
        if not 0 < params.connectivity <= 1:
            raise ValueError("connectivity must be in (0,1]")
        if not 0 <= params.inhibitory_fraction <= 1:
            raise ValueError("inhibitory_fraction must be in [0,1]")
        self.w_in, self.w_res = self._make_weights() if w_in is None or w_res is None else (
            np.asarray(w_in, dtype=np.float32),
            np.asarray(w_res, dtype=np.int8),
        )
        self._validate_weights()

    def _make_weights(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        rng = np.random.default_rng(p.seed)
        w_in = rng.choice(
            np.asarray(p.input_weight_values, dtype=np.float32),
            size=p.neurons,
            p=np.asarray(p.input_weight_probabilities, dtype=np.float64),
        ).astype(np.float32)
        mask = rng.random((p.neurons, p.neurons)) < p.connectivity
        np.fill_diagonal(mask, False)
        signs = rng.choice(
            np.asarray([-1, 1], dtype=np.int8),
            size=(p.neurons, p.neurons),
            p=[p.inhibitory_fraction, 1.0 - p.inhibitory_fraction],
        )
        w_res = (mask * signs).astype(np.int8)
        return w_in, w_res

    def _validate_weights(self) -> None:
        p = self.params
        if self.w_in.shape != (p.neurons,):
            raise ValueError(f"w_in shape must be {(p.neurons,)}, got {self.w_in.shape}")
        if self.w_res.shape != (p.neurons, p.neurons):
            raise ValueError(f"w_res shape must be {(p.neurons, p.neurons)}, got {self.w_res.shape}")
        if not np.all(np.isin(self.w_res, (-1, 0, 1))):
            raise ValueError("w_res must be ternary {-1,0,+1}")
        if np.any(np.diag(self.w_res) != 0):
            raise ValueError("Self-connections are disabled; diagonal must be zero")

    @property
    def recurrent_edges(self) -> int:
        return int(np.count_nonzero(self.w_res))

    def transform(self, X: np.ndarray, return_spikes: bool = False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected X shape [batch,time], got {X.shape}")
        batch, timesteps = X.shape
        n = self.params.neurons
        v = np.full((batch, n), self.params.reset_value, dtype=np.float32)
        spikes = np.zeros((batch, n), dtype=np.float32)
        counts = np.zeros((batch, n), dtype=np.uint8)
        history = np.zeros((batch, timesteps, n), dtype=np.uint8) if return_spikes else None

        for t in range(timesteps):
            recurrent = spikes @ self.w_res.T.astype(np.float32)
            v = (
                self.params.leak * v
                + self.params.input_gain * X[:, t, None] * self.w_in[None, :]
                + self.params.recurrent_gain * recurrent
            )
            spikes = (v >= self.params.threshold).astype(np.float32)
            counts += spikes.astype(np.uint8)
            if history is not None:
                history[:, t, :] = spikes.astype(np.uint8)
            v = np.where(spikes > 0, self.params.reset_value, v)

        if return_spikes:
            return counts, history
        return counts

    def save_weights(self, path: str) -> None:
        np.savez_compressed(path, w_in=self.w_in, w_res=self.w_res)
