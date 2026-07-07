"""CLI: fine-tune wav2vec2 for Kazakh ASR (config file + CLI overrides)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.training import TrainConfig, run_training  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="YAML/JSON TrainConfig file.")
    # Common overrides (None => fall back to config/defaults).
    p.add_argument("--dataset")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--base-model", dest="base_model")
    p.add_argument("--epochs", type=float)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--learning-rate", dest="learning_rate", type=float)
    args = p.parse_args()

    cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = cfg.apply_overrides(overrides)

    print("Training configuration:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    run_training(cfg)


if __name__ == "__main__":
    main()
