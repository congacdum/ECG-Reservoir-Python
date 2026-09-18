import pytest
import numpy as np
from src.model import reservoir_brian2
from src.model.reservoir_numpy import NumpyLIFReservoir, ReservoirParams


def test_brian2_backend_availability_contract():
    if reservoir_brian2.BRIAN2_AVAILABLE:
        reservoir_brian2.require_brian2()
    else:
        with pytest.raises(reservoir_brian2.Brian2UnavailableError):
            reservoir_brian2.require_brian2()


@pytest.mark.skipif(not reservoir_brian2.BRIAN2_AVAILABLE, reason="Brian2 is optional")
def test_brian2_smoke_and_sample_reset():
    w_in = np.array([0.5, 1.0, 2.0, 1.0], dtype=np.float32)
    w_res = np.array(
        [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 1], [0, 0, 0, 0]],
        dtype=np.int8,
    )
    X = np.array([[0.4, 0.5, 0.6, 0.7], [1.8, 1.9, 2.0, 1.7]], dtype=np.float32)
    batch = reservoir_brian2.transform_brian2(X, w_in, w_res)
    assert batch.shape == (2, 4)
    assert np.isfinite(batch).all()
    assert np.array_equal(
        reservoir_brian2.transform_brian2(X[1:2], w_in, w_res)[0], batch[1]
    )


@pytest.mark.skipif(not reservoir_brian2.BRIAN2_AVAILABLE, reason="Brian2 is optional")
def test_brian2_controlled_case_matches_numpy_reference():
    w_in = np.array([0.5, 1.0, 2.0, 1.0], dtype=np.float32)
    w_res = np.array(
        [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 1], [0, 0, 0, 0]],
        dtype=np.int8,
    )
    X = np.array(
        [[0.4, 0.5, 0.6, 0.7, 0.8], [1.8, 1.9, 2.0, 1.7, 1.6]],
        dtype=np.float32,
    )
    brian_counts = reservoir_brian2.transform_brian2(X, w_in, w_res, batch_size=2)
    numpy_counts = NumpyLIFReservoir(
        ReservoirParams(neurons=4, seed=42), w_in=w_in, w_res=w_res
    ).transform(X)
    assert np.array_equal(brian_counts, numpy_counts)
