from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class NoBiasRidgeReadout:
    alpha: float = 100.0
    class_balanced_sample_weight: bool = True

    def __post_init__(self) -> None:
        self.model = Ridge(alpha=self.alpha, fit_intercept=False)

    @staticmethod
    def _targets(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.int64)
        return np.where(y == 1, 1.0, -1.0)

    @staticmethod
    def _balanced_weights(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y, minlength=2)
        if np.any(counts == 0):
            raise ValueError("Both classes are required for balanced sample weights")
        return np.where(y == 0, len(y) / (2 * counts[0]), len(y) / (2 * counts[1]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NoBiasRidgeReadout":
        sample_weight = self._balanced_weights(y) if self.class_balanced_sample_weight else None
        self.model.fit(np.asarray(X), self._targets(y), sample_weight=sample_weight)
        if np.any(np.asarray(self.model.intercept_) != 0):
            raise AssertionError("No-bias readout unexpectedly learned a non-zero intercept")
        return self

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.model.coef_, dtype=np.float64).reshape(-1)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(X)), dtype=np.float64).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) >= 0).astype(np.int64)
