"""Standard error-rate metrics (WER / CER) with a self-contained aligner.

``jiwer`` is used when available for the headline WER number, but we also ship a
dependency-free Levenshtein aligner because the morphology-aware metric
(:mod:`narkesken.metrics.morph_wer`) needs the *alignment*, not just the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Op(str, Enum):
    MATCH = "match"
    SUB = "sub"
    INS = "ins"
    DEL = "del"


@dataclass
class AlignOp:
    op: Op
    ref: str | None
    hyp: str | None


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[AlignOp]:
    """Levenshtein alignment with backtrace. Returns the edit script.

    Uses unit costs (the classic WER model). O(|ref|·|hyp|) time and memory.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost, # sub / match
            )

    # Backtrace.
    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(AlignOp(Op.MATCH, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(AlignOp(Op.SUB, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(AlignOp(Op.DEL, ref[i - 1], None))
            i -= 1
        else:
            ops.append(AlignOp(Op.INS, None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus WER. Uses ``jiwer`` if installed, else the built-in aligner."""
    try:
        import jiwer

        return float(jiwer.wer(references, hypotheses))
    except Exception:
        errors, total = 0, 0
        for ref, hyp in zip(references, hypotheses):
            r, h = ref.split(), hyp.split()
            ops = align(r, h)
            errors += sum(1 for o in ops if o.op != Op.MATCH)
            total += len(r)
        return errors / total if total else 0.0


def character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    errors, total = 0, 0
    for ref, hyp in zip(references, hypotheses):
        ops = align(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        errors += sum(1 for o in ops if o.op != Op.MATCH)
        total += len(ref.replace(" ", ""))
    return errors / total if total else 0.0
