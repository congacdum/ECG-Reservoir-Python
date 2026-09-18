from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _finish(path: str | Path, title: str | None = None) -> None:
    if title:
        plt.title(title)
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_class_distribution(y, path: str | Path) -> None:
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(5, 4))
    plt.bar([str(v) for v in labels], counts)
    plt.xlabel("Class")
    plt.ylabel("Rows")
    for i, count in enumerate(counts):
        plt.text(i, count, str(int(count)), ha="center", va="bottom")
    _finish(path, "Class distribution")


def plot_mean_waveform_by_class(X, y, path: str | Path) -> None:
    X, y = np.asarray(X), np.asarray(y)
    plt.figure(figsize=(7, 4))
    for label in (0, 1):
        subset = X[y == label]
        plt.plot(np.arange(1, X.shape[1] + 1), subset.mean(axis=0), label=f"Class {label}")
    plt.xlabel("ECG sample index")
    plt.ylabel("Signal value")
    plt.legend()
    _finish(path, "Mean waveform by class")


def plot_sample_waveforms(X, y, path: str | Path, per_class: int = 4) -> None:
    X, y = np.asarray(X), np.asarray(y)
    plt.figure(figsize=(8, 5))
    for label in (0, 1):
        idx = np.flatnonzero(y == label)[:per_class]
        for row in idx:
            plt.plot(np.arange(1, X.shape[1] + 1), X[row], alpha=0.65, label=f"Class {label}" if row == idx[0] else None)
    plt.xlabel("ECG sample index")
    plt.ylabel("Signal value")
    plt.legend()
    _finish(path, "Example ECG windows")


def plot_spike_raster(spike_history, path: str | Path, sample_index: int = 0) -> None:
    spikes = np.asarray(spike_history)
    if spikes.ndim == 3:
        spikes = spikes[sample_index]
    if spikes.ndim != 2:
        raise ValueError("Expected [time,neurons] or [batch,time,neurons]")
    t, n = np.nonzero(spikes)
    plt.figure(figsize=(8, 5))
    plt.scatter(t + 1, n, s=8)
    plt.xlabel("Input timestep")
    plt.ylabel("Neuron index")
    _finish(path, "Reservoir spike raster")


def plot_spike_count_distribution(features, path: str | Path) -> None:
    f = np.asarray(features)
    plt.figure(figsize=(7, 4))
    plt.hist(f.ravel(), bins=np.arange(-0.5, f.max() + 1.5, 1))
    plt.xlabel("Spike count per neuron/window")
    plt.ylabel("Frequency")
    _finish(path, "Spike-count distribution")


def plot_confusion_matrix(y_true, y_pred, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], ax=ax, colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_score, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(y_true, y_score, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Precision–recall curve")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(y_true, y_score, path: str | Path) -> None:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    plt.figure(figsize=(7, 4))
    for label in (0, 1):
        plt.hist(y_score[y_true == label], bins=40, alpha=0.6, label=f"Class {label}")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Readout score")
    plt.ylabel("Frequency")
    plt.legend()
    _finish(path, "Readout score distribution")


def plot_metric_comparison(rows: list[dict], path: str | Path, metric: str = "balanced_accuracy") -> None:
    labels = [str(r["name"]) for r in rows]
    values = [float(r[metric]) for r in rows]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylim(max(0, min(values) - 0.1), 1.0)
    plt.ylabel(metric.replace("_", " ").title())
    for i, value in enumerate(values):
        plt.text(i, value, f"{value:.3f}", ha="center", va="bottom")
    _finish(path, f"{metric.replace('_', ' ').title()} comparison")
