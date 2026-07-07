"""CLI: evaluate a model (or a hyp/ref file pair) with WER + morphology-aware M-WER.

Usage
-----
    # Score two text files (one utterance per line):
    python scripts/evaluate.py --hyp hyps.txt --ref refs.txt

    # Run a trained model over a prepared dataset's test split, then score:
    python scripts/evaluate.py --model checkpoints/narkesken --dataset data/prepared \
        --strategy beam+lm --lm checkpoints/kaz.lm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.metrics import (  # noqa: E402
    MorphologyAwareWER,
    character_error_rate,
    word_error_rate,
)
from narkesken.text import normalize_text  # noqa: E402


def _read_lines(path: Path) -> list[str]:
    return [normalize_text(l) for l in path.read_text(encoding="utf-8").splitlines()]


def _run_model(model_dir: Path, dataset_dir: Path, strategy: str,
               lm_path: Path | None, word_lm: bool):
    from datasets import load_from_disk

    from narkesken.inference import Transcriber

    tr = Transcriber(model_dir, lm_path=lm_path, morpheme_lm=not word_lm)
    ds = load_from_disk(str(dataset_dir))
    test = ds["test"] if "test" in ds else ds
    hyps, refs = [], []
    for ex in test:
        audio = ex["audio"]
        hyps.append(tr.transcribe(audio["array"], sampling_rate=audio["sampling_rate"],
                                  strategy=strategy))
        refs.append(ex["text"])
    return hyps, refs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--hyp", type=Path)
    p.add_argument("--ref", type=Path)
    p.add_argument("--strategy", default="greedy",
                   choices=["greedy", "beam", "beam+lm"])
    p.add_argument("--lm", type=Path, help="n-gram LM for beam+lm.")
    p.add_argument("--word-lm", action="store_true", help="LM is word-level, not morpheme.")
    p.add_argument("--w-root", type=float, default=1.0)
    p.add_argument("--w-suffix", type=float, default=0.35)
    args = p.parse_args()

    if args.hyp and args.ref:
        hyps, refs = _read_lines(args.hyp), _read_lines(args.ref)
    elif args.model and args.dataset:
        hyps, refs = _run_model(args.model, args.dataset, args.strategy,
                                args.lm, args.word_lm)
    else:
        raise SystemExit("Provide (--model and --dataset) or (--hyp and --ref).")

    if len(hyps) != len(refs):
        raise SystemExit(f"count mismatch: {len(hyps)} hyp vs {len(refs)} ref")

    wer = word_error_rate(refs, hyps)
    cer = character_error_rate(refs, hyps)
    m = MorphologyAwareWER(w_root=args.w_root, w_suffix=args.w_suffix).compute(refs, hyps)

    print("=" * 60)
    print(f"Utterances        : {len(refs)}")
    print(f"WER               : {wer:.4f}")
    print(f"CER               : {cer:.4f}")
    print("-" * 60)
    print("Morphology-aware error rate (M-WER):")
    print(f"  M-WER           : {m.m_wer:.4f}")
    print(f"  WER (same align): {m.wer_equivalent:.4f}")
    print(f"  exact           : {m.exact}")
    print(f"  suffix-only err : {m.suffix_only}  (root right, inflection wrong)")
    print(f"  root errors     : {m.root_errors}")
    print(f"  insertions/del  : {m.insertions}/{m.deletions}")
    if m.examples:
        print("  sample suffix-only errors:")
        for ex in m.examples:
            print(f"    - {ex}")
    print("=" * 60)


if __name__ == "__main__":
    main()
