"""A pure-Python n-gram language model with modified Kneser-Ney smoothing.

Used for *shallow fusion* during CTC beam-search decoding
(:mod:`narkesken.decoding.beam_search`). For a low-resource, agglutinative
language a word-level LM suffers badly from out-of-vocabulary inflected forms,
so this model can operate at the level of **morphemes** (via
:mod:`narkesken.morphology`) instead of whole words — a sequence such as
``мектеп PL POSS CASE`` factorizes the sparse word ``мектептерімізде`` into
pieces the LM has actually seen, which is the standard trick for LM smoothing
in Turkic ASR.

The estimator is *interpolated modified Kneser-Ney* (Chen & Goodman, 1998):

    p_KN(w | h) = max(c(hw) - D_k, 0) / c(h)  +  γ(h) · p_KN(w | h')

with three discount parameters D_1, D_2, D_3+ estimated from the count-of-counts
n_1..n_4, and continuation counts used for all lower orders.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


@dataclass
class ModifiedKneserNeyLM:
    order: int = 3
    # per-order n-gram counts: counts[n][ngram_tuple] = count
    counts: list[dict] = field(default_factory=list)
    # continuation counts for lower orders (number of distinct left contexts)
    cont_counts: list[dict] = field(default_factory=list)
    discounts: list[tuple[float, float, float]] = field(default_factory=list)
    vocab: set = field(default_factory=set)

    # -- training -----------------------------------------------------------
    @classmethod
    def train(cls, sentences: Iterable[Sequence[str]], order: int = 3) -> "ModifiedKneserNeyLM":
        lm = cls(order=order)
        lm.counts = [defaultdict(int) for _ in range(order + 1)]
        lm.cont_counts = [defaultdict(set) for _ in range(order + 1)]

        for sent in sentences:
            padded = [BOS] * (order - 1) + list(sent) + [EOS]
            lm.vocab.update(sent)
            for n in range(1, order + 1):
                for i in range(len(padded) - n + 1):
                    gram = tuple(padded[i : i + n])
                    lm.counts[n][gram] += 1
                    if n >= 2:
                        # continuation: distinct word preceding this (n-1)-gram
                        suffix = gram[1:]
                        lm.cont_counts[n - 1][suffix].add(gram[0])
        lm.vocab.update({BOS, EOS, UNK})
        lm._estimate_discounts()
        return lm

    def _estimate_discounts(self) -> None:
        """Estimate D_1, D_2, D_3+ per order from counts-of-counts (Chen-Goodman)."""
        self.discounts = [(0.0, 0.0, 0.0)]  # index 0 unused
        for n in range(1, self.order + 1):
            cc = Counter(self.counts[n].values())
            n1, n2, n3, n4 = cc.get(1, 0), cc.get(2, 0), cc.get(3, 0), cc.get(4, 0)
            if n1 == 0:  # degenerate; fall back to a flat absolute discount
                self.discounts.append((0.5, 0.5, 0.5))
                continue
            Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.0
            d1 = 1 - 2 * Y * (n2 / n1) if n1 else 0.0
            d2 = 2 - 3 * Y * (n3 / n2) if n2 else d1
            d3 = 3 - 4 * Y * (n4 / n3) if n3 else d2
            # clamp to [0, k]
            d1 = min(max(d1, 0.0), 1.0)
            d2 = min(max(d2, 0.0), 2.0)
            d3 = min(max(d3, 0.0), 3.0)
            self.discounts.append((d1, d2, d3))

    @staticmethod
    def _discount_for(count: int, discounts: tuple[float, float, float]) -> float:
        if count <= 0:
            return 0.0
        if count == 1:
            return discounts[0]
        if count == 2:
            return discounts[1]
        return discounts[2]

    # -- scoring ------------------------------------------------------------
    def _cont_prob(self, word: str) -> float:
        """Unigram continuation probability p_cont(word) (lowest order)."""
        # number of distinct bigram types ending in `word`, over total bigram types
        num = len(self.cont_counts[1].get((word,), ()))
        denom = sum(len(v) for v in self.cont_counts[1].values()) or 1
        if num == 0:
            # tiny floor for unseen words, spread over vocab
            return 1.0 / (denom + len(self.vocab) + 1)
        return num / denom

    def prob(self, word: str, context: Sequence[str]) -> float:
        """Interpolated modified-KN probability p(word | context)."""
        context = tuple(context)[-(self.order - 1):]
        return self._prob_recursive(word, context)

    def _prob_recursive(self, word: str, context: tuple) -> float:
        n = len(context) + 1
        if n == 1 or n > self.order:
            return self._cont_prob(word if word in self.vocab else UNK)

        gram = (*context, word)
        # For the highest order use raw counts; for lower orders use continuation.
        if n == self.order:
            ctx_count = self.counts[n - 1].get(context, 0)
            gram_count = self.counts[n].get(gram, 0)
        else:
            ctx_count = sum(len(v) for k, v in self.cont_counts[n].items()
                            if k[:-1] == context) or self.counts[n - 1].get(context, 0)
            gram_count = len(self.cont_counts[n].get(gram[1:], ())) if gram[1:] else 0
            gram_count = self.counts[n].get(gram, gram_count)

        if ctx_count == 0:
            return self._prob_recursive(word, context[1:])

        d1, d2, d3 = self.discounts[n]
        D = self._discount_for(gram_count, (d1, d2, d3))
        higher = max(gram_count - D, 0.0) / ctx_count

        # back-off weight γ(context): mass reserved for lower order
        # γ = (D1·N1 + D2·N2 + D3+·N3+) / ctx_count
        n1 = n2 = n3 = 0
        for g, c in self.counts[n].items():
            if g[:-1] == context:
                if c == 1:
                    n1 += 1
                elif c == 2:
                    n2 += 1
                else:
                    n3 += 1
        gamma = (d1 * n1 + d2 * n2 + d3 * n3) / ctx_count
        lower = self._prob_recursive(word, context[1:])
        return higher + gamma * lower

    def logprob(self, word: str, context: Sequence[str]) -> float:
        p = self.prob(word, context)
        return math.log(p) if p > 0 else -30.0  # floor to avoid -inf

    def score_sentence(self, sent: Sequence[str]) -> float:
        """Total log-prob of a token sequence under the model."""
        padded = [BOS] * (self.order - 1) + list(sent) + [EOS]
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            ctx = padded[i - (self.order - 1): i]
            total += self.logprob(padded[i], ctx)
        return total

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        total_lp, total_n = 0.0, 0
        for s in sentences:
            total_lp += self.score_sentence(s)
            total_n += len(s) + 1  # +1 for EOS
        if total_n == 0:
            return float("inf")
        return math.exp(-total_lp / total_n)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        # defaultdicts/sets don't pickle cleanly across versions; plain-ify them.
        blob = {
            "order": self.order,
            "counts": [dict(c) for c in self.counts],
            "cont_counts": [{k: list(v) for k, v in cc.items()} for cc in self.cont_counts],
            "discounts": self.discounts,
            "vocab": list(self.vocab),
        }
        Path(path).write_bytes(pickle.dumps(blob))

    @classmethod
    def load(cls, path: str | Path) -> "ModifiedKneserNeyLM":
        blob = pickle.loads(Path(path).read_bytes())
        lm = cls(order=blob["order"])
        lm.counts = [defaultdict(int, c) for c in blob["counts"]]
        lm.cont_counts = [defaultdict(set, {k: set(v) for k, v in cc.items()})
                          for cc in blob["cont_counts"]]
        lm.discounts = [tuple(d) for d in blob["discounts"]]
        lm.vocab = set(blob["vocab"])
        return lm
