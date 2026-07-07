"""A rule-based Kazakh morphological analyzer (surface -> root + morpheme chain).

This is a light-weight, dependency-free finite-state-style analyzer. It is *not*
a full morphological transducer (for production use you would compile an FST
such as ``apertium-kaz``); instead it implements the same conditioning logic —
morpheme ordering plus phonological allomorph licensing — as an explicit search,
which keeps it hackable and makes the root/suffix decomposition available to the
morphology-aware evaluation metric without a heavyweight dependency.

Algorithm
---------
For a surface word ``w`` we search for the most plausible parse

    w = root · s_k · s_{k-1} · ... · s_1

where each ``s_i`` fills a distinct, order-respecting inflectional slot. We
consider two morpheme chains — the *nominal* chain (PLURAL, POSSESSIVE, CASE)
and the *verbal* chain (NEGATION, TENSE, PERSON) — and peel suffixes from the
end of the word, one slot at a time, in reverse canonical order.

Each candidate parse is scored:

    score = Σ_i  license(s_i)  −  λ_len · |root_deficit|  −  λ_seg · k

``license`` rewards allomorphs whose vowel harmony and consonant assimilation
agree with the preceding stem (the key signal a flat suffix list lacks);
``λ_seg`` penalizes over-segmentation so we don't shave a bare root down to
noise; ``λ_len`` mildly prefers parses that leave a plausibly-sized root.

The best-scoring parse is returned; ``analyze`` can also return the full ranked
beam for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import suffixes as S
from .phonology import VOWELS


@dataclass
class Morpheme:
    surface: str
    tag: str
    licensed: bool


@dataclass
class Analysis:
    root: str
    # Morphemes are stored innermost-first (root-adjacent suffix at index 0),
    # which is the natural reading order for a gloss: root + s_1 + ... + s_k.
    morphemes: list[Morpheme] = field(default_factory=list)
    chain: str = "nominal"
    score: float = 0.0

    @property
    def suffix_chain(self) -> list[str]:
        """Surface suffixes from inner to outer (root-adjacent first)."""
        return [m.surface for m in self.morphemes]

    @property
    def tags(self) -> list[str]:
        """Morphosyntactic tags in canonical order (root-adjacent first)."""
        return [m.tag for m in self.morphemes]

    def gloss(self) -> str:
        """Human-readable gloss, e.g. ``мектеп+PL+POSS+CASE``."""
        return "+".join([self.root, *self.tags]) if self.morphemes else self.root


class KazakhSegmenter:
    """Greedy-beam morphological analyzer over the nominal & verbal chains."""

    def __init__(
        self,
        *,
        min_root_len: int = 2,
        lambda_seg: float = 0.35,
        lambda_len: float = 0.08,
        unlicensed_penalty: float = 0.9,
        licensed_bonus: float = 1.0,
        beam_size: int = 8,
    ) -> None:
        self.min_root_len = min_root_len
        self.lambda_seg = lambda_seg
        self.lambda_len = lambda_len
        self.unlicensed_penalty = unlicensed_penalty
        self.licensed_bonus = licensed_bonus
        self.beam_size = beam_size

    # -- public API ---------------------------------------------------------
    def analyze(self, word: str, *, top_k: int = 1) -> list[Analysis]:
        """Return up to ``top_k`` ranked analyses (best first)."""
        word = word.strip().lower()
        candidates: list[Analysis] = []
        for chain_name, chain in (("nominal", S.NOMINAL_CHAIN), ("verbal", S.VERBAL_CHAIN)):
            candidates.extend(self._search_chain(word, chain, chain_name))
        # The zero-suffix parse (whole word is a root) is always a fallback.
        candidates.append(Analysis(root=word, morphemes=[], chain="root",
                                   score=self._root_only_score(word)))
        candidates.sort(key=lambda a: a.score, reverse=True)
        # De-duplicate identical (root, tags) parses, keep best.
        seen: set[tuple] = set()
        unique: list[Analysis] = []
        for a in candidates:
            key = (a.root, tuple(a.tags))
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique[:top_k]

    def best(self, word: str) -> Analysis:
        return self.analyze(word, top_k=1)[0]

    def root_of(self, word: str) -> str:
        return self.best(word).root

    # -- internals ----------------------------------------------------------
    def _root_only_score(self, word: str) -> float:
        # Small baseline; longer "roots" that contain a vowel are more plausible.
        has_vowel = any(c in VOWELS for c in word)
        return (0.15 if has_vowel else -0.2) - self.lambda_len * self._root_deficit(word)

    def _root_deficit(self, root: str) -> float:
        """Penalty component: unusually short or vowelless roots are suspicious."""
        deficit = max(0, self.min_root_len + 1 - len(root))
        if not any(c in VOWELS for c in root):
            deficit += 2
        return float(deficit)

    def _search_chain(self, word: str, chain, chain_name: str) -> list[Analysis]:
        """Beam search peeling suffixes from the END, reverse canonical order.

        ``chain`` is ordered root->outer; we consume it in reverse (outer->root),
        so a word ...root+PL+POSS+CASE is peeled CASE, then POSS, then PL.
        """
        # Each beam item: (remaining_stem, morphemes_collected_outer_to_inner, score)
        beam: list[tuple[str, list[Morpheme], float]] = [(word, [], 0.0)]
        completed: list[Analysis] = []

        for paradigm in reversed(chain):
            next_beam: list[tuple[str, list[Morpheme], float]] = []
            for stem, morphs, score in beam:
                # Option 1: skip this (optional) slot entirely.
                next_beam.append((stem, morphs, score))
                # Option 2: try to peel one allomorph of this paradigm.
                for new_stem, surf, licensed in paradigm.match_at_end(stem):
                    if len(new_stem) < self.min_root_len:
                        continue
                    delta = (self.licensed_bonus if licensed
                             else -self.unlicensed_penalty)
                    delta -= self.lambda_seg  # cost per segmentation step
                    m = Morpheme(surface=surf, tag=paradigm.tag, licensed=licensed)
                    next_beam.append((new_stem, [m, *morphs], score + delta))
            # prune
            next_beam.sort(key=lambda t: t[2], reverse=True)
            beam = next_beam[: self.beam_size]

        for stem, morphs, score in beam:
            if not morphs:
                continue  # the all-skip parse is handled as root-only elsewhere
            final = score - self.lambda_len * self._root_deficit(stem)
            completed.append(Analysis(root=stem, morphemes=morphs,
                                      chain=chain_name, score=final))
        return completed


# Module-level convenience instance + helpers.
_DEFAULT = KazakhSegmenter()


def segment(word: str) -> Analysis:
    """Best analysis of ``word`` using the default segmenter."""
    return _DEFAULT.best(word)


def root_suffix(word: str) -> tuple[str, list[str]]:
    """Return ``(root, [suffixes inner->outer])`` — the metric-facing view."""
    a = _DEFAULT.best(word)
    return a.root, a.suffix_chain
