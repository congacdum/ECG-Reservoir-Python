from pathlib import Path
import numpy as np

from src.data.loader import dataframe_to_xy, load_dataframe
from src.model.reservoir_numpy import NumpyLIFReservoir, ReservoirParams

ROOT = Path(__file__).resolve().parents[1]


def test_sample_isolation_reset_semantics():
    df = load_dataframe(ROOT / "data" / "splits" / "train.csv").iloc[:2]
    X, _ = dataframe_to_xy(df)
    reservoir = NumpyLIFReservoir(ReservoirParams(seed=42))
    batch_features = reservoir.transform(X)
    second_alone = reservoir.transform(X[1:2])[0]
    assert np.array_equal(batch_features[1], second_alone)
