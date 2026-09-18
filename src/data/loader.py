from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def parse_sequence(value: str, expected_length: int = 20) -> np.ndarray:
    arr = np.fromstring(str(value), sep=" ", dtype=np.float32)
    if arr.size != expected_length:
        raise ValueError(f"Expected {expected_length} values, got {arr.size}: {value!r}")
    if not np.isfinite(arr).all():
        raise ValueError("Sequence contains NaN or Inf")
    return arr


def load_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "value_sequence",
        "finalLabel",
        "original_index",
        "original_len",
        "generated_window_id",
        "new_len",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def dataframe_to_xy(df: pd.DataFrame, sequence_length: int = 20) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([parse_sequence(v, sequence_length) for v in df["value_sequence"].to_numpy()])
    y = df["finalLabel"].to_numpy(dtype=np.int64)
    return X, y
