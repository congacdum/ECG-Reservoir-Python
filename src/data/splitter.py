from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .checks import (
    audit_dataframe,
    check_no_exact_sequence_overlap,
    check_no_group_overlap,
)


def split_by_group(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
    group_column: str = "original_index",
) -> dict[str, pd.DataFrame]:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1, got {total}")

    audit_dataframe(df)
    group_table = (
        df.groupby(group_column, as_index=False)
        .agg(finalLabel=("finalLabel", "first"))
        .sort_values(group_column)
    )

    train_groups, temp_groups = train_test_split(
        group_table,
        test_size=1.0 - train_fraction,
        random_state=random_state,
        stratify=group_table["finalLabel"],
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test,
        random_state=random_state,
        stratify=temp_groups["finalLabel"],
    )

    ids = {
        "train": set(train_groups[group_column].tolist()),
        "val": set(val_groups[group_column].tolist()),
        "test": set(test_groups[group_column].tolist()),
    }
    splits = {
        name: df[df[group_column].isin(group_ids)].copy().reset_index(drop=True)
        for name, group_ids in ids.items()
    }
    check_no_group_overlap(splits, group_column)
    check_no_exact_sequence_overlap(splits)
    return splits


def write_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
