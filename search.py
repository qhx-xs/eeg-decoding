from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

from eeg_pipeline.splits import build_cv_splits
from eeg_pipeline.training import load_pt_dataset, run_cross_validation, train_fold


def candidate_config(trial: optuna.Trial, base: dict[str, Any], search_epochs: int) -> dict[str, Any]:
    config = dict(base)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
    config.update(
        {
            "max_epochs": search_epochs,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "lstm_hidden_size": trial.suggest_categorical(
                "lstm_hidden_size", [16, 32, 64, 128]
            ),
            "lstm_layers": lstm_layers,
            "lstm_dropout": (
                trial.suggest_float("lstm_dropout", 0.1, 0.6, step=0.1)
                if lstm_layers > 1
                else 0.0
            ),
            "classifier_dropout": trial.suggest_float(
                "classifier_dropout", 0.1, 0.6, step=0.1
            ),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05),
        }
    )
    return config


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
    dataset = load_pt_dataset(dataset_path)
    splits = build_cv_splits(dataset, cv=cv, task=task)
    requested_device = str(base_config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_dir / 'study.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=f"{task}_{cv}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(base_config["seed"]), multivariate=True),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = candidate_config(trial, base_config, search_epochs)
        fold_scores = []
        print(f"\n##### Optuna trial {trial.number} / params={trial.params} #####", flush=True)
        for split in splits:
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
            fold_scores.append(float(result["best_validation_phase_macro_f1"]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        score = float(np.mean(fold_scores))
        trial.set_user_attr("validation_fold_macro_f1", fold_scores)
        print(
            f"trial={trial.number} mean_validation_macro_f1={score:.4f} "
            f"folds={[round(value, 4) for value in fold_scores]}",
            flush=True,
        )
        return score

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, trials - completed_trials)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    best_config = dict(base_config)
    best_config.update(study.best_params)
    best_config["max_epochs"] = final_epochs
    best_config["run_name"] = "optuna_best"
    best_payload = {
        "task": task,
        "cv": cv,
        "study_trials": len(study.trials),
        "search_epochs": search_epochs,
        "final_epochs": final_epochs,
        "best_validation_macro_f1": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "best_params": study.best_params,
        "best_config": best_config,
    }
    (output_dir / "best_config.json").write_text(
        json.dumps(best_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "search_summary.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trials_payload = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
    (output_dir / "trials.json").write_text(
        json.dumps(trials_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n##### Best hyperparameters #####")
    print(json.dumps(best_payload, ensure_ascii=False, indent=2), flush=True)

    if not skip_final:
        final_root = output_dir / "final"
        run_cross_validation(dataset_path, task, cv, best_config, final_root)
    return best_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna global search using validation Macro-F1")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("four_class", "imagery_binary"))
    parser.add_argument("--cv", default="run", choices=("run", "trial"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/search"))
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
