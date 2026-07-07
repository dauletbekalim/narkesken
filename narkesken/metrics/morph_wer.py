"""Morphology-aware error rate (M-WER) for agglutinative-language ASR.

Motivation
----------
Plain WER counts every non-matching word as one full error. For Kazakh that is
badly miscalibrated: a hypothesis that gets the *root* right but attaches the
wrong *suffix chain* (a case/tense/possession slip) is penalized exactly like a
completely unrelated word. Yet those two failures mean very different things —
the first is "understood the word, mis-inflected it", the second is "did not
recognize the word at all" — with very different downstream cost (e.g. for a
reading tutor used by schoolchildren).

M-WER fixes this by making the *substitution* cost a function of the
morphological relationship between the reference and hypothesis words. It reuses
the Levenshtein *alignment* from :mod:`narkesken.metrics.wer` (so insertions and
deletions are handled the standard way) and, for each aligned substitution,
decomposes both words with :mod:`narkesken.morphology` and charges:

    cost(sub) =  w_root · [root mismatch]
              +  w_suffix · (normalized suffix-chain edit distance)

A pure suffix error therefore costs ``w_suffix`` (small); a root error costs at
least ``w_root`` (large). With ``w_root = 1`` and ``w_suffix = 0`` M-WER reduces
to a lenient metric that ignores inflection entirely; with ``w_root = w_suffix =
1`` and a coarse decomposition it approaches ordinary WER. The default sits in
between.

The metric also returns a **breakdown** (exact / suffix-only / root errors,
plus insertions and deletions) so error analysis can see *where* a model fails,
not just how often.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..morphology import KazakhSegmenter
from .wer import Op, align


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class MorphWERResult:
    m_wer: float
    wer_equivalent: float          # unit-cost WER over the same alignment
    n_ref_words: int
    exact: int = 0
    suffix_only: int = 0           # root correct, suffix chain differs
    root_errors: int = 0           # substitutions with a wrong root
    insertions: int = 0
    deletions: int = 0
    total_cost: float = 0.0
    examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"M-WER={self.m_wer:.4f}  (WER={self.wer_equivalent:.4f})  "
            f"exact={self.exact} suffix-only={self.suffix_only} "
            f"root-err={self.root_errors} ins={self.insertions} del={self.deletions}"
        )


class MorphologyAwareWER:
    """Callable scorer for the morphology-aware error rate."""

    def __init__(
        self,
        segmenter: KazakhSegmenter | None = None,
        *,
        w_root: float = 1.0,
        w_suffix: float = 0.35,
        insertion_cost: float = 1.0,
        deletion_cost: float = 1.0,
        collect_examples: int = 12,
    ) -> None:
        self.seg = segmenter or KazakhSegmenter()
        self.w_root = w_root
        self.w_suffix = w_suffix
        self.insertion_cost = insertion_cost
        self.deletion_cost = deletion_cost
        self.collect_examples = collect_examples

    def _sub_cost(self, ref_word: str, hyp_word: str) -> tuple[float, str]:
        """Weighted cost of substituting ``hyp_word`` for ``ref_word``.

        Returns ``(cost, kind)`` where ``kind`` is ``"suffix"`` or ``"root"``.
        """
        ra = self.seg.best(ref_word)
        ha = self.seg.best(hyp_word)
        if ra.root == ha.root:
            # Same root -> charge only for the suffix-chain divergence,
            # normalized by the longer chain so a single wrong marker on a long
            # chain isn't over-counted.
            chain_dist = _levenshtein(ra.suffix_chain, ha.suffix_chain)
            denom = max(len(ra.suffix_chain), len(ha.suffix_chain), 1)
            cost = self.w_suffix * (chain_dist / denom)
            return cost, "suffix"
        # Different root: full root penalty, plus a share of suffix divergence
        # (a wrong root usually drags the suffixes with it).
        chain_dist = _levenshtein(ra.suffix_chain, ha.suffix_chain)
        denom = max(len(ra.suffix_chain), len(ha.suffix_chain), 1)
        cost = self.w_root + self.w_suffix * (chain_dist / denom)
        return cost, "root"

    def score_pair(self, reference: str, hypothesis: str, result: MorphWERResult) -> None:
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        result.n_ref_words += len(ref_words)
        ops = align(ref_words, hyp_words)
        for o in ops:
            if o.op == Op.MATCH:
                result.exact += 1
            elif o.op == Op.INS:
                result.insertions += 1
                result.total_cost += self.insertion_cost
            elif o.op == Op.DEL:
                result.deletions += 1
                result.total_cost += self.deletion_cost
            elif o.op == Op.SUB:
                cost, kind = self._sub_cost(o.ref, o.hyp)
                result.total_cost += cost
                if kind == "suffix":
                    result.suffix_only += 1
                else:
                    result.root_errors += 1
                if len(result.examples) < self.collect_examples and kind == "suffix":
                    result.examples.append(
                        f"{o.ref!r} -> {o.hyp!r}  (root ok, suffix Δ, cost={cost:.2f})"
                    )

    def compute(self, references: list[str], hypotheses: list[str]) -> MorphWERResult:
        if len(references) != len(hypotheses):
            raise ValueError("references and hypotheses must be the same length")
        res = MorphWERResult(m_wer=0.0, wer_equivalent=0.0, n_ref_words=0)
        unit_errors = 0
        for ref, hyp in zip(references, hypotheses):
            before = (res.suffix_only + res.root_errors + res.insertions + res.deletions)
            self.score_pair(ref, hyp, res)
            after = (res.suffix_only + res.root_errors + res.insertions + res.deletions)
            unit_errors += (after - before)
        n = res.n_ref_words or 1
        res.m_wer = res.total_cost / n
        res.wer_equivalent = unit_errors / n
        return res


def morphology_aware_wer(
    references: list[str],
    hypotheses: list[str],
    *,
    w_root: float = 1.0,
    w_suffix: float = 0.35,
) -> MorphWERResult:
    """Convenience wrapper around :class:`MorphologyAwareWER`."""
    return MorphologyAwareWER(w_root=w_root, w_suffix=w_suffix).compute(
        references, hypotheses
    )
