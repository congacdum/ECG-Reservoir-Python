from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "splits" / "train.csv"
VAL_FILE = ROOT / "data" / "splits" / "val.csv"
ANALYSIS_DIR = ROOT / "outputs" / "analysis"


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    values = np.asarray(
        [[float(value) for value in text.split()] for text in frame["value_sequence"].astype(str)],
        dtype=np.float32,
    )
    labels = frame["finalLabel"].to_numpy(dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 20:
        raise ValueError(f"Expected [samples, 20] values in {path}, got {values.shape}")
    return values, labels


def load_train_validation() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = load_split(TRAIN_FILE)
    x_val, y_val = load_split(VAL_FILE)
    return x_train, y_train, x_val, y_val


def feature_sets(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "RAW20": x,
        "RAW20_DIFF19": np.concatenate((x, np.diff(x, axis=1)), axis=1),
    }


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=2)
    return np.where(y == 0, len(y) / (2 * counts[0]), len(y) / (2 * counts[1]))


def score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    pred = (score >= 0).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


def ensure_analysis_dir() -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR

