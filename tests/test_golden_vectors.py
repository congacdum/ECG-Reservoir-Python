import numpy as np

from src.hardware.fixed_point import FixedPointFormat
from src.hardware.golden_vectors import (
    generate_golden_vectors,
    load_golden_vectors,
    write_golden_vectors,
)
from src.hardware.hardware_model import HardwareParams, IntegerLIFReservoir, integer_predict


def _fixture():
    w_in = np.array([0.5, 1.0, 2.0, 1.0], dtype=np.float32)
    w_res = np.array(
        [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 1], [0, 0, 0, 0]],
        dtype=np.int8,
    )
    params = HardwareParams(input_format=FixedPointFormat(12, 10, signed=True))
    reservoir = IntegerLIFReservoir(w_in, w_res, params)
    X = np.array(
        [
            np.linspace(0.40, 0.80, 20, dtype=np.float32),
            np.linspace(1.60, 2.00, 20, dtype=np.float32),
        ]
    )
    labels = np.array([0, 1], dtype=np.int64)
    weights = np.array([7, -9, 11, -13], dtype=np.int8)
    return reservoir, X, labels, weights


def test_trace_and_final_outputs_match_direct_hardware_model():
    reservoir, X, labels, weights = _fixture()
    vectors = generate_golden_vectors(
        reservoir,
        X,
        labels,
        weights,
        ["train_0000", "validation_0000"],
        ["train", "validation"],
    )
    direct_counts = reservoir.transform(X)
    direct_pred, direct_score = integer_predict(direct_counts, weights)
    assert np.array_equal(vectors["spike_counts"], direct_counts)
    assert np.array_equal(vectors["signed_score"], direct_score)
    assert np.array_equal(vectors["predicted_class"], direct_pred)
    assert np.array_equal(vectors["spike_count_state"][:, -1, :], direct_counts)


def test_vectors_are_deterministic_legal_and_lossless(tmp_path):
    reservoir, X, labels, weights = _fixture()
    kwargs = {
        "reservoir": reservoir,
        "X": X,
        "labels": labels,
        "w_out_int8": weights,
        "sample_ids": ["train_0000", "validation_0000"],
        "source_splits": ["train", "validation"],
    }
    first = generate_golden_vectors(**kwargs)
    second = generate_golden_vectors(**kwargs)
    for name in first:
        assert np.array_equal(first[name], second[name]), name

    manifest_a = write_golden_vectors(tmp_path / "a", first, reservoir, 20, 42)
    manifest_b = write_golden_vectors(tmp_path / "b", second, reservoir, 20, 42)
    assert manifest_a["model_sha256"] == manifest_b["model_sha256"]
    _, loaded = load_golden_vectors(tmp_path / "a")
    for name, value in first.items():
        assert np.array_equal(loaded[name], value), name

    p = reservoir.params
    assert first["input_quantized"].min() >= p.input_format.min_int
    assert first["input_quantized"].max() <= p.input_format.max_int
    assert first["membrane_after_update"].min() >= -(1 << (p.membrane_bits - 1))
    assert first["membrane_after_update"].max() <= (1 << (p.membrane_bits - 1)) - 1
    assert set(np.unique(first["threshold_result"])).issubset({0, 1})
    assert int(first["spike_counts"].min()) >= 0
    assert int(first["spike_counts"].max()) <= 20
    assert int(first["w_out_int8"].min()) >= -128
    assert int(first["w_out_int8"].max()) <= 127
    assert int(first["signed_score"].min()) >= -(1 << 19)
    assert int(first["signed_score"].max()) <= (1 << 19) - 1
    assert manifest_a["no_bias"] == {
        "reservoir_bias_current": False,
        "readout_intercept": False,
        "exported_bias": False,
    }
