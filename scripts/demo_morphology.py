"""Dependency-free demo of the morphology analyzer + M-WER metric.

Runs with only the Python standard library (no torch/transformers), so it's the
quickest way to see the interesting parts of Narkesken working:

    python scripts/demo_morphology.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.metrics import MorphologyAwareWER, word_error_rate  # noqa: E402
from narkesken.morphology import KazakhSegmenter  # noqa: E402

WORDS = [
    "мектеп",           # bare root
    "мектептер",        # + PL
    "мектептерім",      # + PL + POSS.1sg
    "мектептерімде",    # + PL + POSS + LOC
    "балаларына",       # bala + PL + POSS + DAT (epenthetic -н-)
    "үйлерде",          # üy + PL + LOC
    "досым",            # dos + POSS.1sg
    "барамын",          # bar + PRES + 1sg
    "оқыдым",           # oqy + PAST + 1sg
    "келмедім",         # kel + NEG + PAST + 1sg
]


def demo_segmentation() -> None:
    seg = KazakhSegmenter()
    print("Morphological analysis (surface -> root + gloss):")
    print("-" * 60)
    for w in WORDS:
        a = seg.best(w)
        print(f"  {w:<16} ->  {a.gloss():<34} [{a.chain}, score={a.score:.2f}]")
    print()


def demo_metric() -> None:
    # References vs hypotheses that differ ONLY by inflection (suffix) in some
    # words, and by the actual root in others.
    refs = [
        "мен мектепке барамын",         # ref
        "балалар кітаптарын оқыды",
        "ол үйге келді",
    ]
    hyps = [
        "мен мектепте барамын",         # мектепКЕ -> мектепТЕ : suffix-only (root ok)
        "балалар кітаптарын оқыды",     # exact
        "ол үйге кетті",                # келді -> кетті : different root
    ]
    wer = word_error_rate(refs, hyps)
    m = MorphologyAwareWER(w_root=1.0, w_suffix=0.35).compute(refs, hyps)

    print("Standard WER vs morphology-aware M-WER on the same alignment:")
    print("-" * 60)
    print(f"  WER   = {wer:.4f}   (мектепте and кетті each count as 1 full error)")
    print(f"  M-WER = {m.m_wer:.4f}")
    print(f"    {m.summary()}")
    if m.examples:
        print("    suffix-only cases the metric discounted:")
        for ex in m.examples:
            print(f"      - {ex}")
    print()
    print("Takeaway: WER blames the model equally for a wrong CASE marker and a")
    print("wrong verb; M-WER charges the case slip far less than the lexical error.")


if __name__ == "__main__":
    demo_segmentation()
    demo_metric()
