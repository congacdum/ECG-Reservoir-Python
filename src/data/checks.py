from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .loader import dataframe_to_xy, load_dataframe


def check_sequence_length(df: pd.DataFrame, expected_length: int = 20) -> None:
    X, _ = dataframe_to_xy(df, expected_length)
    if X.shape[1] != expected_length:
        raise AssertionError("Unexpected sequence length")


def check_no_nan_inf(df: pd.DataFrame, expected_length: int = 20) -> None:
    X, _ = dataframe_to_xy(df, expected_length)
    if not np.isfinite(X).all():
        raise AssertionError("NaN or Inf found in input sequences")


def check_binary_labels(df: pd.DataFrame) -> None:
    labels = set(df["finalLabel"].unique().tolist())
    if not labels.issubset({0, 1}) or not labels:
        raise AssertionError(f"Expected binary labels {{0,1}}, got {labels}")


def check_group_label_consistency(df: pd.DataFrame, group_column: str = "original_index") -> None:
    counts = df.groupby(group_column)["finalLabel"].nunique()
    bad = counts[counts != 1]
    if len(bad):
        raise AssertionError(f"Groups with conflicting labels: {bad.index.tolist()[:10]}")


def check_no_group_overlap(splits: Mapping[str, pd.DataFrame], group_column: str = "original_index") -> None:
    keys = list(splits)
    groups = {k: set(splits[k][group_column].tolist()) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlap = groups[a] & groups[b]
            if overlap:
                raise AssertionError(f"Group leakage between {a} and {b}: {len(overlap)} overlapping groups")


def _sequence_hashes(df: pd.DataFrame, expected_length: int = 20) -> set[str]:
    X, _ = dataframe_to_xy(df, expected_length)
    return {hashlib.sha256(row.astype(np.float32).tobytes()).hexdigest() for row in X}


def check_no_exact_sequence_overlap(splits: Mapping[str, pd.DataFrame], expected_length: int = 20) -> None:
    keys = list(splits)
    hashes = {k: _sequence_hashes(splits[k], expected_length) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlap = hashes[a] & hashes[b]
            if overlap:
                raise AssertionError(f"Exact waveform leakage between {a} and {b}: {len(overlap)} sequences")


def audit_dataframe(df: pd.DataFrame, expected_length: int = 20) -> dict:
    check_sequence_length(df, expected_length)
    check_no_nan_inf(df, expected_length)
    check_binary_labels(df)
    check_group_label_consistency(df)
    return {
        "rows": int(len(df)),
        "groups": int(df["original_index"].nunique()),
        "class_rows": {str(k): int(v) for k, v in df["finalLabel"].value_counts().sort_index().items()},
        "class_groups": {
            str(k): int(v)
            for k, v in df.groupby("original_index")["finalLabel"].first().value_counts().sort_index().items()
        },
        "sequence_length": expected_length,
        "new_len_values": sorted(int(v) for v in df["new_len"].unique()),
    }


def audit_split_files(paths: Mapping[str, str | Path], expected_length: int = 20) -> dict:
    splits = {name: load_dataframe(path) for name, path in paths.items()}
    for df in splits.values():
        audit_dataframe(df, expected_length)
    check_no_group_overlap(splits)
    check_no_exact_sequence_overlap(splits, expected_length)
    return {
        name: audit_dataframe(df, expected_length) for name, df in splits.items()
    }
