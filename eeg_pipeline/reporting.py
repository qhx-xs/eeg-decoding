from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalized(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=denominator != 0,
    )


def _write_csv(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])


def _plot_matrix(
    path: Path,
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    normalized = _normalized(matrix)
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(class_names) * 3.0), 4.8))
    for axis, values, subtitle, value_format in (
        (axes[0], matrix, "Counts", "d"),
        (axes[1], normalized, "Row-normalized", ".2f"),
    ):
        image = axis.imshow(values, interpolation="nearest", cmap="Blues", vmin=0)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set(
            title=subtitle,
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        axis.tick_params(axis="x", rotation=35)
        threshold = float(values.max()) / 2.0 if values.size else 0.0
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                text = format(int(value), value_format) if value_format == "d" else format(value, value_format)
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_history(path: Path, folds: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fold in folds:
        history = fold["history"]
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["train_loss"] for item in history], label=f"fold {fold['fold']}")
        axes[1].plot(
            epochs,
            [item["val_phase_macro_f1"] for item in history],
            label=f"fold {fold['fold']}",
        )
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].set(title="Validation phase/trial Macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_cv_artifacts(report: dict[str, Any], output_dir: Path, class_names: list[str]) -> None:
    """Save human-readable confusion matrices and learning curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in ("phase_or_trial", "window"):
        matrices = [
            np.asarray(fold["metrics"]["test"][level]["confusion_matrix"], dtype=np.int64)
            for fold in report["folds"]
        ]
        aggregate = np.sum(matrices, axis=0)
        _write_csv(output_dir / f"confusion_{level}_counts.csv", aggregate, class_names)
        _write_csv(output_dir / f"confusion_{level}_normalized.csv", _normalized(aggregate), class_names)
        _plot_matrix(
            output_dir / f"confusion_{level}.png",
            aggregate,
            class_names,
            f"{report['task']} / {report['cv']} / {level} (all folds)",
        )
        for fold, matrix in zip(report["folds"], matrices):
            _plot_matrix(
                output_dir / f"fold_{fold['fold']}_confusion_{level}.png",
                matrix,
                class_names,
                f"{report['task']} / {report['cv']} / fold {fold['fold']} / {level}",
            )
    _plot_history(output_dir / "learning_curves.png", report["folds"])
