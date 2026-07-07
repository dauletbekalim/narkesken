"""Evaluate a fine-tuned Narkesken model.

Two metrics are reported:

1. **Standard WER** (Word Error Rate) via ``jiwer`` — the usual ASR yardstick.

2. **Morphology-aware error rate** (a STUB / starting point) — the idea proposed
   in the README: for an agglutinative language, a wrong *suffix* on a correct
   *root* is a milder error than getting the root wrong, but plain WER scores
   them identically. This function makes that distinction explicit in code.

    # Evaluate a trained model on a prepared dataset:
    python evaluate.py --model checkpoints/demo --dataset data/prepared

    # Or score a hypotheses/references file pair without running a model:
    python evaluate.py --hyp hyps.txt --ref refs.txt

⚠️  The morphology-aware metric here is a deliberately simple heuristic, NOT a
validated linguistic model. Proper Kazakh morphological segmentation is its own
research problem; this is a runnable seed for that work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Morphology-aware scoring (STUB)
# ---------------------------------------------------------------------------

# A crude, illustrative list of common Kazakh suffixes, longest-first so that
# greedy stripping peels the outermost suffix first. This is NOT a complete or
# linguistically rigorous inventory — it exists to demonstrate the metric.
# Kazakh suffixation is governed by vowel harmony and consonant assimilation,
# so a real segmenter would use a morphological analyzer (e.g. an FST/Apertium
# model), not a flat list.
_COMMON_SUFFIXES = sorted(
    [
        # plural
        "лар", "лер", "дар", "дер", "тар", "тер",
        # possessive
        "ым", "ім", "ың", "ің", "сы", "сі", "мыз", "міз", "ңыз", "ңіз",
        # case (genitive/dative/accusative/locative/ablative/instrumental)
        "ның", "нің", "дың", "дің", "тың", "тің",
        "ға", "ге", "қа", "ке", "на", "не",
        "ны", "ні", "ды", "ді", "ты", "ті",
        "да", "де", "та", "те",
        "дан", "ден", "тан", "тен", "нан", "нен",
        "мен", "бен", "пен",
        # verbal / tense / person (a small sample)
        "ды", "ді", "ты", "ті", "ған", "ген", "қан", "кен",
        "мын", "мін", "сың", "сің", "ады", "еді",
    ],
    key=len,
    reverse=True,
)


def segment_root_suffix(word: str, max_suffixes: int = 3) -> tuple[str, list[str]]:
    """Greedily split a word into (root, [suffix chain]).

    Repeatedly strips the longest matching suffix from the end, up to
    ``max_suffixes`` times, while keeping the remaining root non-trivial.

    Returns the root and the list of stripped suffixes in outer-to-inner order.
    This is a heuristic approximation — see the module docstring.
    """
    suffixes: list[str] = []
    root = word
    for _ in range(max_suffixes):
        stripped = False
        for suf in _COMMON_SUFFIXES:
            # Require the root to stay at least 2 chars so we don't peel a word
            # down to nothing.
            if len(root) - len(suf) >= 2 and root.endswith(suf):
                root = root[: -len(suf)]
                suffixes.append(suf)
                stripped = True
                break
        if not stripped:
            break
    return root, suffixes


@dataclass
class MorphScore:
    morph_error_rate: float
    n_words: int
    n_correct: int          # fully correct words
    n_suffix_only_errors: int  # root correct, suffix chain wrong
    n_root_errors: int      # root wrong (or word missing/inserted)


def morphology_aware_error_rate(
    predictions: list[str],
    references: list[str],
    *,
    root_weight: float = 1.0,
    suffix_weight: float = 0.4,
) -> MorphScore:
    """A morphology-aware alternative to WER (STUB).

    For each reference/hypothesis pair, words are aligned positionally (a simple
    proxy — a production version would use edit-distance alignment). Each aligned
    word contributes:

        * 0.0            if the word matches exactly,
        * ``suffix_weight`` if the ROOT matches but the suffix chain differs,
        * ``root_weight``   if the root differs (or a word is missing/extra).

    The returned rate is total penalty / number of reference words. With
    ``suffix_weight < root_weight`` this rewards a model that "hears the word"
    but mis-inflects it — the behavior plain WER hides.
    """
    total_penalty = 0.0
    n_words = 0
    n_correct = 0
    n_suffix_only = 0
    n_root_errors = 0

    for hyp, ref in zip(predictions, references):
        ref_words = ref.split()
        hyp_words = hyp.split()
        n_words += len(ref_words)

        for i, ref_w in enumerate(ref_words):
            if i >= len(hyp_words):
                # Missing word -> treated as a root-level error.
                total_penalty += root_weight
                n_root_errors += 1
                continue

            hyp_w = hyp_words[i]
            if hyp_w == ref_w:
                n_correct += 1
                continue

            ref_root, _ = segment_root_suffix(ref_w)
            hyp_root, _ = segment_root_suffix(hyp_w)
            if ref_root == hyp_root:
                total_penalty += suffix_weight
                n_suffix_only += 1
            else:
                total_penalty += root_weight
                n_root_errors += 1

        # Extra hypothesis words (insertions) count as root-level errors.
        if len(hyp_words) > len(ref_words):
            extra = len(hyp_words) - len(ref_words)
            total_penalty += root_weight * extra
            n_root_errors += extra

    rate = total_penalty / n_words if n_words else 0.0
    return MorphScore(
        morph_error_rate=rate,
        n_words=n_words,
        n_correct=n_correct,
        n_suffix_only_errors=n_suffix_only,
        n_root_errors=n_root_errors,
    )


# ---------------------------------------------------------------------------
# Standard WER + optional model inference
# ---------------------------------------------------------------------------


def compute_wer(predictions: list[str], references: list[str]) -> float:
    import jiwer

    return jiwer.wer(references, predictions)


def transcribe_dataset(model_dir: Path, dataset_dir: Path) -> tuple[list[str], list[str]]:
    """Run a fine-tuned wav2vec2 model over the test split; return (hyps, refs)."""
    import torch
    from datasets import load_from_disk
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    model.eval()

    ds = load_from_disk(str(dataset_dir))
    test = ds["test"] if "test" in ds else ds

    hyps: list[str] = []
    refs: list[str] = []
    for example in test:
        audio = example["audio"]
        inputs = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyps.append(processor.batch_decode(pred_ids)[0])
        refs.append(example["text"])
    return hyps, refs


def _read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Fine-tuned model directory.")
    parser.add_argument("--dataset", type=Path, help="Prepared dataset directory.")
    parser.add_argument("--hyp", type=Path, help="Hypotheses file (one per line).")
    parser.add_argument("--ref", type=Path, help="References file (one per line).")
    parser.add_argument("--suffix-weight", type=float, default=0.4)
    parser.add_argument("--root-weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.hyp and args.ref:
        hyps = _read_lines(args.hyp)
        refs = _read_lines(args.ref)
    elif args.model and args.dataset:
        hyps, refs = transcribe_dataset(args.model, args.dataset)
    else:
        raise SystemExit("Provide either (--model and --dataset) or (--hyp and --ref).")

    if len(hyps) != len(refs):
        raise SystemExit(f"hyp/ref count mismatch: {len(hyps)} vs {len(refs)}")

    wer = compute_wer(hyps, refs)
    morph = morphology_aware_error_rate(
        hyps, refs, root_weight=args.root_weight, suffix_weight=args.suffix_weight
    )

    print("=" * 56)
    print(f"Utterances evaluated : {len(refs)}")
    print(f"Standard WER         : {wer:.4f}")
    print("-" * 56)
    print("Morphology-aware error rate (STUB — illustrative):")
    print(f"  morph error rate   : {morph.morph_error_rate:.4f}")
    print(f"  words              : {morph.n_words}")
    print(f"  fully correct      : {morph.n_correct}")
    print(f"  suffix-only errors : {morph.n_suffix_only_errors}  "
          f"(root right, inflection wrong)")
    print(f"  root errors        : {morph.n_root_errors}")
    print("=" * 56)


if __name__ == "__main__":
    main()
