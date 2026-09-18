from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_development_scripts_do_not_open_test_csv():
    for script_name in ("02_train.py", "03_validate.py", "04_quantize.py"):
        text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert 'test.csv' not in text, f"{script_name} must remain blind to test.csv"
