from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"


def test_analysis_artifacts_are_present_and_train_validation_only():
    required = [
        "current_baseline.md",
        "baseline_metrics.csv",
        "baseline_summary.md",
        "window_statistics.csv",
        "class_feature_statistics.csv",
        "simple_rule_metrics.csv",
        "baseline_feature_importance.csv",
        "dataset_provenance.md",
        "recurrence_ablation.csv",
        "reservoir_svd.csv",
        "reservoir_size_sweep.csv",
        "reservoir_seed_stability.csv",
        "robustness_metrics.csv",
        "model_v2_recommendation.md",
        "environment_reference.txt",
        "requirements-lock.txt",
    ]
    for name in required:
        assert (ANALYSIS / name).exists(), name
    metrics = pd.read_csv(ANALYSIS / "baseline_metrics.csv")
    assert {"RAW20", "RAW20_DIFF19", "64 spike-count features"} <= set(metrics["feature_set"])
    assert {"HistGradientBoosting", "DecisionTree_depth5"} <= set(metrics["model"])
    assert not metrics.empty
    ablation = pd.read_csv(ANALYSIS / "recurrence_ablation.csv")
    assert set(ablation["recurrence"]) == {"ON_locked_artifact", "OFF_Wres_zero"}
    assert float(ablation.loc[ablation["recurrence"] == "ON_locked_artifact", "val_balanced_accuracy"].iloc[0]) > float(
        ablation.loc[ablation["recurrence"] == "OFF_Wres_zero", "val_balanced_accuracy"].iloc[0]
    )
    seed_summary = json.loads((ANALYSIS / "reservoir_seed_stability_summary.json").read_text(encoding="utf-8"))
    assert seed_summary["seed_count"] == 10
    assert seed_summary["test_used"] is False
    robustness = json.loads((ANALYSIS / "robustness_summary.json").read_text(encoding="utf-8"))
    assert robustness["test_used"] is False

