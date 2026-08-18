from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


def make_trial_fold_assignment(
    trials: list[dict[str, Any]], n_splits: int = 5, random_state: int = 42
) -> dict[int, int]:
    """Assign whole trials to folds, stratified by recording and object."""
    if len({int(trial["trial_id"]) for trial in trials}) != len(trials):
        raise ValueError("trial_id values must be unique")
    strata = np.array(
        [f"{int(trial['run_id'])}:{int(trial['object_label'])}" for trial in trials]
    )
    counts = {label: int(np.sum(strata == label)) for label in np.unique(strata)}
    if min(counts.values(), default=0) < n_splits:
        raise ValueError(f"Each run*object stratum needs at least {n_splits} trials; got {counts}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    assignment: dict[int, int] = {}
    placeholder = np.zeros(len(trials), dtype=np.uint8)
    for fold, (_, test_indices) in enumerate(splitter.split(placeholder, strata)):
        for index in test_indices:
            assignment[int(trials[index]["trial_id"])] = fold
    return assignment


def _assert_disjoint_groups(dataset: dict, splits: dict[str, torch.Tensor], group_key: str) -> None:
    group_sets = {
        name: set(dataset[group_key][indices].tolist()) for name, indices in splits.items()
    }
    names = list(group_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = group_sets[left].intersection(group_sets[right])
            if overlap:
                raise AssertionError(f"{group_key} leakage between {left} and {right}: {sorted(overlap)}")


def build_cv_splits(dataset: dict, cv: str, task: str) -> list[dict[str, torch.Tensor | int]]:
    """Build deterministic train/validation/test indices without trial leakage."""
    if cv not in {"run", "trial"}:
        raise ValueError("cv must be 'run' or 'trial'")
    if task not in {"four_class", "imagery_binary"}:
        raise ValueError("task must be 'four_class' or 'imagery_binary'")
    eligible = torch.ones(len(dataset["features"]), dtype=torch.bool)
    if task == "imagery_binary":
        eligible &= dataset["phase"] == 1

    fold_values = dataset["run_id"] if cv == "run" else dataset["trial_fold"]
    fold_count = 4 if cv == "run" else 5
    output = []
    for test_fold in range(fold_count):
        val_fold = (test_fold + 1) % fold_count
        test_mask = eligible & (fold_values == test_fold)
        val_mask = eligible & (fold_values == val_fold)
        train_mask = eligible & (fold_values != test_fold) & (fold_values != val_fold)
        split = {
            "fold": test_fold,
            "train": torch.nonzero(train_mask, as_tuple=False).flatten(),
            "val": torch.nonzero(val_mask, as_tuple=False).flatten(),
            "test": torch.nonzero(test_mask, as_tuple=False).flatten(),
        }
        _assert_disjoint_groups(dataset, {key: split[key] for key in ("train", "val", "test")}, "trial_id")
        if cv == "run":
            _assert_disjoint_groups(dataset, {key: split[key] for key in ("train", "val", "test")}, "run_id")
        if any(len(split[name]) == 0 for name in ("train", "val", "test")):
            raise ValueError(f"Empty subset in {cv} fold {test_fold}")
        output.append(split)
    return output
