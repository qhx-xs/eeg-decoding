from __future__ import annotations

import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from EEG_Model import EEG_CNN_BiLSTM_Attention

from .splits import build_cv_splits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pt_dataset(path: Path) -> dict[str, Any]:
    try:
        dataset = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        dataset = torch.load(path, map_location="cpu")
    required = {
        "features",
        "label_four",
        "label_object",
        "phase",
        "run_id",
        "trial_id",
        "trial_fold",
        "event_code",
        "qc_flag",
        "metadata",
    }
    missing = required.difference(dataset)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    features = dataset["features"]
    if features.ndim != 4 or tuple(features.shape[1:]) != (8, 200, 6):
        raise ValueError(f"Expected features [N,8,200,6], got {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("Dataset contains non-finite features")
    return dataset


def fit_standardizer(features: torch.Tensor, train_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    train = features[train_indices].float()
    mean = train.mean(dim=(0, 2), keepdim=True)
    std = train.std(dim=(0, 2), keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def class_weights(labels: torch.Tensor, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels[indices], minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError(f"Training fold is missing a class: counts={counts.tolist()}")
    return labels[indices].numel() / (num_classes * counts)


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "sample_count": int(len(y_true)),
    }


def metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    window_predictions = logits.argmax(dim=1)
    window_metrics = _metric_block(labels.numpy(), window_predictions.numpy(), num_classes)

    aggregated_logits = []
    aggregated_labels = []
    for group_id in torch.unique(group_ids, sorted=True):
        mask = group_ids == group_id
        group_labels = torch.unique(labels[mask])
        if len(group_labels) != 1:
            raise ValueError(f"Aggregation group {int(group_id)} contains multiple labels")
        aggregated_logits.append(logits[mask].mean(dim=0))
        aggregated_labels.append(group_labels[0])
    phase_logits = torch.stack(aggregated_logits)
    phase_labels = torch.stack(aggregated_labels)
    phase_predictions = phase_logits.argmax(dim=1)
    phase_metrics = _metric_block(phase_labels.numpy(), phase_predictions.numpy(), num_classes)
    return {"phase_or_trial": phase_metrics, "window": window_metrics}


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    dataset: dict[str, Any],
    task: str,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits_parts = []
    label_parts = []
    index_parts = []
    with torch.no_grad():
        for samples, labels, indices in loader:
            logits_parts.append(model(samples.to(device)).cpu())
            label_parts.append(labels)
            index_parts.append(indices)
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    indices = torch.cat(index_parts)
    if task == "four_class":
        group_ids = dataset["trial_id"][indices] * 100 + dataset["event_code"][indices]
    else:
        group_ids = dataset["trial_id"][indices]
    return metrics_from_logits(logits, labels, group_ids, num_classes)


def _make_loader(
    normalized_features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = TensorDataset(normalized_features[indices], labels[indices], indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_fold(
    dataset: dict[str, Any],
    split: dict[str, torch.Tensor | int],
    task: str,
    config: dict[str, Any],
    output_dir: Path | None,
    device: torch.device,
    *,
    save_artifacts: bool = True,
    evaluate_test: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    fold = int(split["fold"])
    seed = int(config["seed"]) + fold
    seed_everything(seed)
    labels = dataset["label_four"] if task == "four_class" else dataset["label_object"]
    num_classes = 4 if task == "four_class" else 2
    train_indices = split["train"]
    mean, std = fit_standardizer(dataset["features"], train_indices)
    normalized = (dataset["features"].float() - mean) / std

    batch_size = int(config["batch_size"])
    loaders = {
        name: _make_loader(normalized, labels, split[name], batch_size, name == "train")
        for name in ("train", "val", "test")
    }
    model = EEG_CNN_BiLSTM_Attention(
        channels=dataset["features"].shape[1],
        num_classes=num_classes,
        lstm_hidden_size=int(config["lstm_hidden_size"]),
        lstm_layers=int(config["lstm_layers"]),
        lstm_dropout=float(config["lstm_dropout"]),
        classifier_dropout=float(config["classifier_dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(labels, train_indices, num_classes).to(device),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["max_epochs"]),
        eta_min=float(config.get("minimum_learning_rate", 1e-6)),
    )

    best_score = -1.0
    best_epoch = -1
    best_state = None
    history = []
    training_started = time.perf_counter()
    if verbose:
        print(
            f"\n=== fold {fold} | task={task} | epochs={config['max_epochs']} | "
            f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} ===",
            flush=True,
        )
    for epoch in range(int(config["max_epochs"])):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_samples = 0
        for samples, target, _ in loaders["train"]:
            samples = samples.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(samples)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            total_samples += len(target)

        val_metrics = _evaluate(model, loaders["val"], dataset, task, num_classes, device)
        val_score = float(val_metrics["phase_or_trial"]["macro_f1"])
        scheduler.step()
        average_loss = total_loss / total_samples
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": average_loss,
                "val_phase_macro_f1": val_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        log_every = max(1, int(config.get("log_every_epochs", 10)))
        if verbose and ((epoch + 1) == 1 or (epoch + 1) % log_every == 0 or (epoch + 1) == int(config["max_epochs"])):
            print(
                f"fold={fold} epoch={epoch + 1:03d}/{config['max_epochs']} "
                f"loss={average_loss:.4f} val_macro_f1={val_score:.4f} "
                f"best={best_score:.4f}@{best_epoch} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"time={history[-1]['seconds']:.2f}s",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metric_names = ("train", "val", "test") if evaluate_test else ("train", "val")
    split_metrics = {
        name: _evaluate(model, loaders[name], dataset, task, num_classes, device)
        for name in metric_names
    }
    qc_counts = {
        name: int(dataset["qc_flag"][split[name]].sum().item()) for name in ("train", "val", "test")
    }
    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_validation_phase_macro_f1": best_score,
        "epochs_ran": len(history),
        "training_seconds": time.perf_counter() - training_started,
        "metrics": split_metrics,
        "qc_flagged_windows": qc_counts,
        "subset_window_counts": {name: int(len(split[name])) for name in ("train", "val", "test")},
        "history": history,
    }
    if save_artifacts:
        if output_dir is None:
            raise ValueError("output_dir is required when save_artifacts=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "normalization_mean": mean,
                "normalization_std": std,
                "task": task,
                "num_classes": num_classes,
                "training_config": dict(config),
                "dataset_format_version": dataset["format_version"],
                "fold": fold,
                "best_epoch": best_epoch,
            },
            output_dir / f"fold_{fold}.pt",
        )
    if verbose:
        print(
            f"fold {fold} complete: trained {len(history)} epochs, "
            f"best validation Macro-F1={best_score:.4f} at epoch {best_epoch}",
            flush=True,
        )
    return result


def _aggregate_cv(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy")
    summary: dict[str, Any] = {}
    for level in ("phase_or_trial", "window"):
        summary[level] = {}
        for metric in metric_names:
            values = np.array([result["metrics"]["test"][level][metric] for result in results])
            summary[level][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return summary


def save_cross_validation_report(
    dataset: dict[str, Any],
    task: str,
    cv: str,
    fold_results: list[dict[str, Any]],
    training_config: dict[str, Any] | list[dict[str, Any]],
    run_dir: Path,
    device: torch.device,
    run_name: str,
) -> dict[str, Any]:
    report = {
        "task": task,
        "cv": cv,
        "device": str(device),
        "run_name": run_name,
        "training_config": training_config,
        "fold_count": len(fold_results),
        "primary_metric_level": "phase_or_trial",
        "folds": fold_results,
        "test_summary": _aggregate_cv(fold_results),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from .reporting import save_cv_artifacts

    class_names = (
        dataset["metadata"]["four_class_names"]
        if task == "four_class"
        else dataset["metadata"]["object_names"]
    )
    save_cv_artifacts(report, run_dir, list(class_names))
    print(f"\nResults and confusion matrices saved to: {run_dir}", flush=True)
    return report


def run_cross_validation(
    dataset_path: Path,
    task: str,
    cv: str,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    dataset = load_pt_dataset(dataset_path)
    splits = build_cv_splits(dataset, cv=cv, task=task)
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; run training on the GPU server")
    device = torch.device(requested_device)
    run_name = str(config.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = output_root / f"{task}_{cv}_{run_name}"
    fold_results = [
        train_fold(dataset, split, task, config, run_dir, device) for split in splits
    ]
    return save_cross_validation_report(
        dataset,
        task,
        cv,
        fold_results,
        dict(config),
        run_dir,
        device,
        run_name,
    )
