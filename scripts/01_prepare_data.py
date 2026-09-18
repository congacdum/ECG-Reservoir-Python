#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataframe
from src.data.splitter import split_by_group, write_splits
from src.data.checks import audit_dataframe, audit_split_files
from src.evaluation.plots import plot_class_distribution, plot_mean_waveform_by_class, plot_sample_waveforms
from src.utils.io import load_yaml, save_json
from src.data.loader import dataframe_to_xy


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic leakage-safe ECG splits.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_path = ROOT / cfg["data"]["raw_file"]
    out_dir = ROOT / "data" / "splits"
    existing = [out_dir / f"{name}.csv" for name in ("train", "val", "test")]
    if any(p.exists() for p in existing) and not args.force:
        raise SystemExit("Split files already exist. Refusing to overwrite; use --force explicitly.")

    df = load_dataframe(raw_path)
    raw_audit = audit_dataframe(df, cfg["data"]["sequence_length"])
    s = cfg["split"]
    splits = split_by_group(
        df,
        train_fraction=s["train"],
        val_fraction=s["validation"],
        test_fraction=s["test"],
        random_state=s["random_state"],
        group_column=cfg["data"]["group_column"],
    )
    write_splits(splits, out_dir)
    split_audit = audit_split_files(
        {name: out_dir / f"{name}.csv" for name in ("train", "val", "test")},
        cfg["data"]["sequence_length"],
    )
    info = {
        "raw": raw_audit,
        "splits": split_audit,
        "policy": "group-wise by original_index; exact waveform overlap checked",
        "sha256": {
            "raw": sha256_file(raw_path),
            **{name: sha256_file(out_dir / f"{name}.csv") for name in ("train", "val", "test")},
        },
    }
    save_json(info, out_dir / "split_info.json")

    # Exploratory figures are based on train only, never test.
    X_train, y_train = dataframe_to_xy(splits["train"], cfg["data"]["sequence_length"])
    fig_dir = ROOT / "outputs" / "runs" / "run_001" / "figures"
    plot_class_distribution(y_train, fig_dir / "train_class_distribution.png")
    plot_mean_waveform_by_class(X_train, y_train, fig_dir / "train_mean_waveform_by_class.png")
    plot_sample_waveforms(X_train, y_train, fig_dir / "train_sample_waveforms.png")

    print("Data preparation complete.")
    for name, audit in split_audit.items():
        print(f"{name}: rows={audit['rows']}, groups={audit['groups']}, class_rows={audit['class_rows']}")


if __name__ == "__main__":
    main()
