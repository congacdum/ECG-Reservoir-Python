from pathlib import Path

from src.data.checks import audit_split_files

ROOT = Path(__file__).resolve().parents[1]


def test_split_files_have_no_group_or_exact_sequence_leakage():
    report = audit_split_files(
        {name: ROOT / "data" / "splits" / f"{name}.csv" for name in ("train", "val", "test")},
        expected_length=20,
    )
    assert report["train"]["groups"] == 6365
    assert report["val"]["groups"] == 1364
    assert report["test"]["groups"] == 1365
