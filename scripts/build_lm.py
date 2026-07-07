"""CLI: train a modified Kneser-Ney n-gram LM for shallow fusion.

By default the LM is trained on **morpheme** sequences (each word segmented via
the Kazakh analyzer), which is the recommended setup for Kazakh — it drastically
reduces OOV compared with a word-level model. Pass ``--word-level`` to train on
raw word tokens instead.

Input is a plain-text corpus, one sentence per line (already Kazakh Cyrillic).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.decoding import ModifiedKneserNeyLM  # noqa: E402
from narkesken.morphology import KazakhSegmenter  # noqa: E402
from narkesken.text import normalize_text  # noqa: E402


def iter_sequences(corpus: Path, *, word_level: bool):
    seg = None if word_level else KazakhSegmenter()
    with corpus.open("r", encoding="utf-8") as f:
        for line in f:
            words = normalize_text(line).split()
            if not words:
                continue
            if word_level:
                yield words
            else:
                toks: list[str] = []
                for w in words:
                    a = seg.best(w)
                    toks.extend([a.root, *a.tags])
                yield toks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path, help="One sentence per line.")
    p.add_argument("--out", required=True, type=Path, help="Output .lm pickle.")
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--word-level", action="store_true")
    args = p.parse_args()

    seqs = list(iter_sequences(args.corpus, word_level=args.word_level))
    if not seqs:
        raise SystemExit(f"No usable sentences in {args.corpus}")
    lm = ModifiedKneserNeyLM.train(seqs, order=args.order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lm.save(args.out)
    ppl = lm.perplexity(seqs)  # train-set perplexity (sanity check only)
    unit = "words" if args.word_level else "morphemes"
    print(f"Trained {args.order}-gram {unit} LM on {len(seqs)} sentences.")
    print(f"  vocab={len(lm.vocab)}  train-set perplexity={ppl:.2f}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
