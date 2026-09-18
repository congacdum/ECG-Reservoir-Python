from __future__ import annotations

from .reservoir_numpy import ReservoirParams


def reservoir_params_from_config(cfg: dict) -> ReservoirParams:
    r = cfg["reservoir"]
    return ReservoirParams(
        neurons=r["neurons"],
        connectivity=r["connectivity"],
        inhibitory_fraction=r["inhibitory_fraction"],
        input_weight_values=tuple(r["input_weight_values"]),
        input_weight_probabilities=tuple(r["input_weight_probabilities"]),
        leak=r["leak_numerator"] / r["leak_denominator"],
        input_gain=r["input_gain_numerator"] / r["input_gain_denominator"],
        recurrent_gain=r["recurrent_gain_numerator"] / r["recurrent_gain_denominator"],
        threshold=r["threshold"],
        reset_value=r["reset_value"],
        seed=cfg["project"]["seed"],
    )
