"""CLI: package .wav + transcript pairs into a HuggingFace datasets directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.data import prepare_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-char-count", type=int, default=1)
    args = p.parse_args()
    prepare_dataset(args.audio_dir, args.out, test_size=args.test_size,
                    seed=args.seed, min_char_count=args.min_char_count)


if __name__ == "__main__":
    main()
