import numpy as np

from src.evaluation.metrics import evaluate_binary_classifier, evaluate_model_health
from src.hardware.fixed_point import FixedPointFormat, quantization_diagnostics
from src.hardware.hardware_model import integer_readout_diagnostics


def test_quantization_diagnostics_measure_clipping_and_underflow():
    fmt = FixedPointFormat(bits=4, fractional_bits=2, signed=True)
    stats = quantization_diagnostics(np.array([-10.0, 0.0, 0.01, 1.0, 3.0]), fmt)
    assert stats["input_clipping_count"] == 2
    assert stats["input_underflow_to_zero_count"] == 1
    assert stats["input_underflow_to_zero_rate"] == 0.25


def test_model_health_reports_gap_and_warnings():
    train = evaluate_binary_classifier([0, 1], [0, 1], [0.0, 1.0])
    validation = evaluate_binary_classifier([0, 1], [0, 0], [0.0, -1.0])
    health = evaluate_model_health(train, validation)
    assert health["generalization_gap"] == 0.5
    assert health["overfitting_warning"] is True
    assert health["underfitting_warning"] is False


def test_readout_diagnostics_measure_saturation():
    stats = integer_readout_diagnostics(np.array([[20, 20], [-20, -20]]), np.array([127, 127]), 8)
    assert stats["readout_accumulator_upper_saturation_count"] == 1
    assert stats["readout_accumulator_lower_saturation_count"] == 1
