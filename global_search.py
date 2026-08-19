from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
import torch

from eeg_pipeline.splits import build_random_split
from eeg_pipeline.training import (
    load_pt_dataset,
    save_cross_validation_report,
    train_fold,
)
from search import candidate_config


def completed_trials(study: optuna.Study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def run_global_search(
    dataset_path: Path,
    task: str,
    base_config: dict[str, Any],
    output_dir: Path,
    trials: int,
    search_epochs: int,
    final_epochs: int,
    repeats: int,
) -> dict[str, Any]:
    dataset = load_pt_dataset(dataset_path)
    if not torch.cuda.is_available() and str(base_config.get("device", "cuda")).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(str(base_config.get("device", "cuda")))
    split = build_random_split(
        dataset,
        task,
        random_state=int(base_config["seed"]),
        train_fraction=float(base_config.get("train_fraction", 0.70)),
        validation_fraction=float(base_config.get("validation_fraction", 0.15)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_dir / 'study.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=f"{task}_random_window_global",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(base_config["seed"])),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = candidate_config(trial, base_config, search_epochs)
        if int(config["rnn_layers"]) == 1:
            config["rnn_dropout"] = 0.0
        result = train_fold(
            dataset,
            split,
            task,
            config,
            output_dir=None,
            device=device,
            save_artifacts=False,
            evaluate_test=False,
            verbose=False,
        )
        score = float(result["best_validation_phase_macro_f1"])
        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        trial.set_user_attr("training_seconds", float(result["training_seconds"]))
        print(f"trial={trial.number} validation_macro_f1={score:.4f} params={trial.params}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score

    remaining = max(0, trials - completed_trials(study))
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    best_config = dict(base_config)
    best_config.update(study.best_params)
    if int(best_config["rnn_layers"]) == 1:
        best_config["rnn_dropout"] = 0.0
    best_config["max_epochs"] = final_epochs
    best_config["run_name"] = "random_global_best_repeated"

    final_dir = output_dir / "final"
    results = []
    configs = []
    for repeat in range(repeats):
        repeated_split = dict(split)
        repeated_split["fold"] = repeat
        repeated_config = dict(best_config)
        repeated_config["seed"] = int(base_config["seed"])
        results.append(train_fold(dataset, repeated_split, task, repeated_config, final_dir, device))
        configs.append(repeated_config)

    report = save_cross_validation_report(
        dataset,
        task,
        "random_window_repeated",
        results,
        configs,
        final_dir,
        device,
        "random_global_best_repeated",
    )
    search_summary = {
        "split": "random_window_70_15_15",
        "warning": "Windows from the same trial may occur in different subsets; metrics may be optimistic.",
        "search_trials": completed_trials(study),
        "search_epochs": search_epochs,
        "final_epochs": final_epochs,
        "final_repeats": repeats,
        "best_trial": int(study.best_trial.number),
        "best_validation_macro_f1": float(study.best_value),
        "best_config": best_config,
        "test_summary": report["test_summary"],
    }
    (output_dir / "search_summary.json").write_text(
        json.dumps(search_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return search_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Global Optuna search and repeated training on one random window split"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("four_class", "imagery_binary"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/global_search"))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--search-epochs", type=int, default=100)
    parser.add_argument("--final-epochs", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.trials, args.search_epochs, args.final_epochs, args.repeats) <= 0:
        parser.error("trial, epoch, and repeat counts must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_global_search(
        args.data.resolve(),
        args.task,
        config,
        args.output.resolve() / args.task,
        args.trials,
        args.search_epochs,
        args.final_epochs,
        args.repeats,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
