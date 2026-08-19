from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_pipeline.training import run_cross_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CNN-GRU-Transformer-Attention")
    parser.add_argument("--data", required=True, type=Path, help="Preprocessed .pt dataset")
    parser.add_argument("--task", required=True, choices=("four_class", "imagery_binary"))
    parser.add_argument("--cv", required=True, choices=("random", "run", "trial"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, help="Override fixed epoch count (early stopping is disabled)")
    parser.add_argument("--run-name", help="Stable suffix for the output directory")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        config["max_epochs"] = args.epochs
    if args.run_name:
        config["run_name"] = args.run_name
    report = run_cross_validation(
        dataset_path=args.data.resolve(),
        task=args.task,
        cv=args.cv,
        config=config,
        output_root=args.output.resolve(),
    )
    print(json.dumps(report["test_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
