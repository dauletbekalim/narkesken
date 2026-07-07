"""CLI: generate synthetic sample data (demo only — not real speech)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.data import make_sample_data  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/sample"))
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make_sample_data(args.out, num=args.num, seed=args.seed)


if __name__ == "__main__":
    main()
