from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_pipeline.preprocessing import preprocess_edf_directory, save_preprocessed


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert EDF+ EEG recordings to six-band PT features")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing EDF files")
    parser.add_argument("--output", required=True, type=Path, help="Output .pt path")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/preprocess.json"), help="Preprocessing JSON config"
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload, summary = preprocess_edf_directory(args.input.resolve(), config)
    summary_path = save_preprocessed(payload, summary, args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved dataset: {args.output.resolve()}")
    print(f"Saved anonymous summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
