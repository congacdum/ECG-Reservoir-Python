from __future__ import annotations

import numpy as np


def spike_count(spike_history: np.ndarray) -> np.ndarray:
    spikes = np.asarray(spike_history)
    if spikes.ndim != 3:
        raise ValueError("Expected spike history [batch,time,neurons]")
    return spikes.sum(axis=1, dtype=np.uint16)
