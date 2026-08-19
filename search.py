from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
import torch

from eeg_pipeline.splits import build_cv_splits
from eeg_pipeline.training import (
    load_pt_dataset,
    save_cross_validation_report,
    train_fold,
)


def candidate_config(trial: optuna.Trial, base: dict[str, Any], epochs: int) -> dict[str, Any]:
    config = dict(base)
    config.update(
        {
            "max_epochs": epochs,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "transformer_d_model": trial.suggest_categorical(
                "transformer_d_model", [64, 128, 256]
            ),
            "transformer_heads": trial.suggest_categorical(
                "transformer_heads", [2, 4, 8]
            ),
            "transformer_layers": trial.suggest_int("transformer_layers", 1, 4),
            "transformer_feedforward": trial.suggest_categorical(
                "transformer_feedforward", [128, 256, 512]
            ),
            "transformer_dropout": trial.suggest_float(
                "transformer_dropout", 0.1, 0.5, step=0.1
            ),
            "classifier_dropout": trial.suggest_float(
                "classifier_dropout", 0.1, 0.6, step=0.1
            ),
            "label_smoothing": trial.suggest_float(
                "label_smoothing", 0.0, 0.2, step=0.05
            ),
        }
    )
    return config


def _completed_trials(study: optuna.Study) -> int:
    return sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)


def _trial_payload(study: optuna.Study) -> list[dict[str, Any]]:
    return [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]


def run_search(
    dataset_path: Path,
    task: str,
    cv: str,
    base_config: dict[str, Any],
    output_dir: Path,
    trials: int,
    search_epochs: int,
    final_epochs: int,
    skip_final: bool,
) -> dict[str, Any]:
    """Tune each outer fold without exposing its test subset."""
    dataset = load_pt_dataset(dataset_path)
    splits = build_cv_splits(dataset, cv=cv, task=task)
    requested_device = str(base_config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_dir / 'study.db').resolve().as_posix()}"
    fold_searches: list[dict[str, Any]] = []
    final_configs: list[dict[str, Any]] = []

    for split in splits:
        fold = int(split["fold"])
        study = optuna.create_study(
            study_name=f"{task}_{cv}_outer_fold_{fold}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(base_config["seed"]) + fold),
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            config = candidate_config(trial, base_config, search_epochs)
            print(
                f"\n##### outer fold {fold} / Optuna trial {trial.number} / "
                f"params={trial.params} #####",
                flush=True,
            )
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
            print(
                f"outer_fold={fold} trial={trial.number} "
                f"validation_macro_f1={score:.4f}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return score

        remaining = max(0, trials - _completed_trials(study))
        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)

        best_config = dict(base_config)
        best_config.update(study.best_params)
        best_config["max_epochs"] = final_epochs
        best_config["run_name"] = "nested_optuna"
        final_configs.append(best_config)
        fold_payload = {
            "fold": fold,
            "completed_trials": _completed_trials(study),
            "best_validation_macro_f1": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "best_params": study.best_params,
            "best_config": best_config,
        }
        fold_searches.append(fold_payload)
        (output_dir / f"fold_{fold}_trials.json").write_text(
            json.dumps(_trial_payload(study), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n##### outer fold {fold} best #####")
        print(json.dumps(fold_payload, ensure_ascii=False, indent=2), flush=True)

    summary = {
        "search_type": "nested_cv",
        "task": task,
        "cv": cv,
        "trials_per_outer_fold": trials,
        "search_epochs": search_epochs,
        "final_epochs": final_epochs,
        "test_data_used_during_search": False,
        "folds": fold_searches,
    }
    (output_dir / "best_configs.json").write_text(
        json.dumps(final_configs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not skip_final:
        final_dir = output_dir / "final" / f"{task}_{cv}_nested_optuna"
        fold_results = [
            train_fold(dataset, split, task, config, final_dir, device)
            for split, config in zip(splits, final_configs)
        ]
        save_cross_validation_report(
            dataset,
            task,
            cv,
            fold_results,
            final_configs,
            final_dir,
            device,
            "nested_optuna",
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nested Optuna search with an untouched outer test subset"
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("four_class", "imagery_binary"))
    parser.add_argument("--cv", default="run", choices=("run", "trial"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/search_nested"))
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--search-epochs", type=int, default=60)
    parser.add_argument("--final-epochs", type=int, default=200)
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    if min(args.trials, args.search_epochs, args.final_epochs) <= 0:
        parser.error("trials and epoch counts must be positive")
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    task_output = args.output.resolve() / f"{args.task}_{args.cv}"
    run_search(
        args.data.resolve(),
        args.task,
        args.cv,
        base_config,
        task_output,
        args.trials,
        args.search_epochs,
        args.final_epochs,
        args.skip_final,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
