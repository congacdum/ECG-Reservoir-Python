from __future__ import annotations

"""Optional Brian2 reference backend.

This module mirrors the locked no-bias reservoir concept and is deliberately
kept separate from the NumPy/fixed-point FPGA reference path. It is runtime
smoke-tested when Brian2 is available, but its continuous-time input scaling
does not yet produce bit-equivalent spike counts to the locked NumPy model.
See PROJECT_KNOWLEDGE.md and CHANGE_LOG.md.
"""

import numpy as np

try:
    from brian2 import (
        NeuronGroup,
        Network,
        SpikeMonitor,
        Synapses,
        TimedArray,
        defaultclock,
        ms,
        prefs,
        start_scope,
    )
    BRIAN2_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent optional dependency
    BRIAN2_AVAILABLE = False


class Brian2UnavailableError(RuntimeError):
    """Raised when the optional Brian2 backend is requested but unavailable."""


def require_brian2() -> None:
    if not BRIAN2_AVAILABLE:
        raise Brian2UnavailableError(
            "Brian2 is not installed. Install requirements.txt in an environment "
            "with package/network access, then rerun the Brian2 backend."
        )


def transform_brian2(
    X: np.ndarray,
    w_in: np.ndarray,
    w_res: np.ndarray,
    *,
    dt_ms: float = 1.0,
    leak_tau_ms: float = 8.0,
    threshold: float = 2.5,
    input_gain: float = 1.5,
    recurrent_gain: float = 1 / 16,
    batch_size: int = 64,
) -> np.ndarray:
    """Return per-neuron spike counts for independent ECG windows.

    The implementation uses direct continuous-valued input through TimedArray,
    fixed recurrent synapses, no additive bias current, and disjoint neuron
    blocks for independent samples. Chunking keeps the state-isolation
    guarantee while making validation-sized runs practical. It is a software
    LSM reference, not the final FPGA execution model.
    """
    require_brian2()
    X = np.asarray(X, dtype=np.float32)
    w_in = np.asarray(w_in, dtype=np.float32)
    w_res = np.asarray(w_res, dtype=np.int8)
    if X.ndim != 2:
        raise ValueError("X must have shape [batch,time]")
    if w_in.ndim != 1:
        raise ValueError("w_in must have shape [neurons]")
    if w_res.shape != (w_in.size, w_in.size):
        raise ValueError("w_res shape mismatch")
    if dt_ms <= 0 or leak_tau_ms <= 0:
        raise ValueError("dt_ms and leak_tau_ms must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = w_in.size
    results = np.zeros((len(X), n), dtype=np.uint8)

    # Brian codegen is kept in numpy mode for portability of the reference path.
    prefs.codegen.target = "numpy"
    defaultclock.dt = dt_ms * ms

    for batch_start in range(0, len(X), batch_size):
        batch = X[batch_start : batch_start + batch_size]
        batch_count = len(batch)
        start_scope()
        stimulus = TimedArray(batch.T, dt=dt_ms * ms)
        sample_ids = np.repeat(np.arange(batch_count, dtype=np.int32), n)
        # The tau/integration_dt factor makes one Euler step apply the direct
        # input current used by the discrete NumPy reference. No bias current.
        eqs = """
        dv/dt = (-v + (tau/integration_dt)*gain*input_weight*stimulus(t, sample_id))/tau : 1
        input_weight : 1
        gain : 1 (constant)
        tau : second (constant)
        integration_dt : second (constant)
        sample_id : integer (constant)
        """
        neurons = NeuronGroup(
            batch_count * n,
            eqs,
            threshold=f"v >= {threshold}",
            reset="v = 0",
            method="euler",
        )
        neurons.v = 0
        neurons.input_weight = np.tile(w_in, batch_count)
        neurons.gain = input_gain
        neurons.tau = leak_tau_ms * ms
        neurons.integration_dt = dt_ms * ms
        neurons.sample_id = sample_ids

        destination, source = np.nonzero(w_res)
        offsets = np.arange(batch_count, dtype=np.int32) * n
        network_objects = [neurons]
        if len(source):
            syn = Synapses(neurons, neurons, model="w : 1", on_pre="v_post += w")
            syn.connect(
                i=np.tile(source, batch_count) + np.repeat(offsets, len(source)),
                j=np.tile(destination, batch_count) + np.repeat(offsets, len(destination)),
            )
            syn.w = recurrent_gain * np.tile(w_res[destination, source], batch_count)
            network_objects.append(syn)
        monitor = SpikeMonitor(neurons, record=False)
        network_objects.append(monitor)
        net = Network(*network_objects)
        net.run(batch.shape[1] * dt_ms * ms)
        results[batch_start : batch_start + batch_count] = np.asarray(
            monitor.count, dtype=np.uint8
        ).reshape(batch_count, n)

    return results
