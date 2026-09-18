from pathlib import Path

from src.model.readout import NoBiasRidgeReadout
from src.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_readout_have_no_bias():
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    assert cfg["reservoir"]["bias"] is False
    assert cfg["readout"]["fit_intercept"] is False
    model = NoBiasRidgeReadout(alpha=1.0)
    assert model.model.fit_intercept is False
