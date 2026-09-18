import numpy as np
from src.evaluation.metrics import evaluate_binary_classifier


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    score = np.array([-2, -1, 1, 3])
    pred = (score >= 0).astype(int)
    m = evaluate_binary_classifier(y, pred, score)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["specificity"] == 1.0
    assert m["sensitivity_recall"] == 1.0
