"""CTC prefix beam search with optional n-gram LM shallow fusion.

Implements the prefix-beam-search decoder of Hannun et al. (2014),
"First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional
Recurrent DNNs", adapted for a character-level wav2vec2 CTC head over the Kazakh
alphabet.

The decoder maintains, for each candidate prefix ℓ, two probabilities:

    p_b(ℓ, t)  — prob. of ℓ where the alignment ends in a *blank*  at frame t
    p_nb(ℓ, t) — prob. of ℓ where the alignment ends in a *non-blank* at frame t

and extends prefixes frame-by-frame, folding repeated characters and blanks per
the CTC collapse rule. A **language-model factor** is applied multiplicatively
(additively in log space) each time a word delimiter completes a word:

    score(ℓ) = log p_ctc(ℓ)  +  α · log p_lm(words(ℓ))  +  β · |words(ℓ)|

α is the LM weight and β a word-insertion bonus that offsets the LM's bias
toward shorter outputs — the standard shallow-fusion objective.

The LM hook is deliberately abstract (``lm_word_logprob``) so it can be a
word-level model or a **morpheme-factored** one (recommended for Kazakh; see
:func:`morpheme_lm_scorer`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

NEG_INF = -float("inf")


def _logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@dataclass
class BeamEntry:
    p_b: float = NEG_INF     # log-prob, prefix ends in blank
    p_nb: float = NEG_INF    # log-prob, prefix ends in non-blank

    @property
    def total(self) -> float:
        return _logsumexp(self.p_b, self.p_nb)


# LM hook: given the completed words so far and the just-finished word, return a
# log-probability. Returns 0.0 for "no opinion" (e.g. no LM).
LMScorer = Callable[[Sequence[str], str], float]


def ctc_prefix_beam_search(
    log_probs,                      # 2-D array-like [T, V] of log-softmax outputs
    id_to_char: dict[int, str],
    *,
    blank_id: int,
    word_delimiter: str = "|",
    beam_width: int = 32,
    lm_scorer: LMScorer | None = None,
    lm_weight: float = 0.4,         # α
    word_bonus: float = 1.2,        # β
    prune_threshold: float = -12.0, # skip tokens with log-prob below this
) -> list[tuple[str, float]]:
    """Decode one utterance. Returns ``[(hypothesis, score), ...]`` best-first.

    ``log_probs`` may be a numpy array or a plain list of lists of floats; only
    indexing and ``len`` are used, so there is no hard numpy dependency here.
    """
    T = len(log_probs)
    V = len(log_probs[0]) if T else 0

    # Beams keyed by prefix string (word-delimiters included as `word_delimiter`).
    beams: dict[str, BeamEntry] = {"": BeamEntry(p_b=0.0, p_nb=NEG_INF)}
    # Cache of accumulated LM score per prefix so we don't rescore whole words.
    lm_cache: dict[str, float] = {"": 0.0}

    def lm_score_of(prefix: str) -> float:
        if lm_scorer is None:
            return 0.0
        if prefix in lm_cache:
            return lm_cache[prefix]
        words = [w for w in prefix.split(word_delimiter) if w]
        if len(words) < 1:
            val = 0.0
        else:
            history, last = words[:-1], words[-1]
            parent = word_delimiter.join(words[:-1])
            base = lm_cache.get(parent, None)
            if base is None:
                base = lm_score_of((parent + word_delimiter) if parent else "")
            val = base + lm_weight * lm_scorer(history, last) + word_bonus
        lm_cache[prefix] = val
        return val

    for t in range(T):
        row = log_probs[t]
        # Only consider the most promising symbols this frame (pruning).
        cand_ids = [c for c in range(V) if row[c] > prune_threshold]
        if blank_id not in cand_ids:
            cand_ids.append(blank_id)

        next_beams: dict[str, BeamEntry] = defaultdict(BeamEntry)

        for prefix, entry in beams.items():
            last_char = prefix[-1] if prefix else ""
            for c in cand_ids:
                p = row[c]
                if c == blank_id:
                    # Blank keeps the prefix, accumulates into p_b.
                    nb = next_beams[prefix]
                    nb.p_b = _logsumexp(nb.p_b, entry.total + p)
                    continue

                ch = id_to_char[c]
                if ch == last_char:
                    # Repeat of the last non-blank char: two cases.
                    # (a) collapse onto same prefix (came via non-blank path)
                    same = next_beams[prefix]
                    same.p_nb = _logsumexp(same.p_nb, entry.p_nb + p)
                    # (b) genuine new emission requires an intervening blank
                    new_prefix = prefix + ch
                    npe = next_beams[new_prefix]
                    npe.p_nb = _logsumexp(npe.p_nb, entry.p_b + p)
                else:
                    new_prefix = prefix + ch
                    npe = next_beams[new_prefix]
                    npe.p_nb = _logsumexp(npe.p_nb, entry.total + p)

        # Rank by CTC total + LM contribution, keep the top `beam_width`.
        scored = sorted(
            next_beams.items(),
            key=lambda kv: kv[1].total + lm_score_of(kv[0]),
            reverse=True,
        )
        beams = dict(scored[:beam_width])

    final = sorted(
        beams.items(),
        key=lambda kv: kv[1].total + lm_score_of(kv[0]),
        reverse=True,
    )
    results: list[tuple[str, float]] = []
    for prefix, entry in final:
        text = prefix.replace(word_delimiter, " ").strip()
        results.append((text, entry.total + lm_score_of(prefix)))
    return results


def greedy_ctc_decode(log_probs, id_to_char: dict[int, str], *, blank_id: int,
                      word_delimiter: str = "|") -> str:
    """Best-path (argmax) CTC decoding — the cheap baseline for comparison."""
    prev = None
    out = []
    for t in range(len(log_probs)):
        row = log_probs[t]
        c = max(range(len(row)), key=lambda i: row[i])
        if c != blank_id and c != prev:
            out.append(id_to_char[c])
        prev = c
    return "".join(out).replace(word_delimiter, " ").strip()


def morpheme_lm_scorer(lm, segmenter) -> LMScorer:
    """Build an LM hook that factors each word into morphemes before scoring.

    ``lm`` is a :class:`~narkesken.decoding.ngram_lm.ModifiedKneserNeyLM` trained
    on morpheme sequences; ``segmenter`` is a
    :class:`~narkesken.morphology.KazakhSegmenter`. Scoring a word means
    segmenting it and summing the LM log-probs of its morphemes in context,
    which sidesteps the OOV explosion of a word-level Kazakh LM.
    """
    def score(history_words: Sequence[str], word: str) -> float:
        # Expand history + current word into a morpheme stream.
        hist_morphs: list[str] = []
        for w in history_words[-4:]:
            a = segmenter.best(w)
            hist_morphs.extend([a.root, *a.tags])
        a = segmenter.best(word)
        cur = [a.root, *a.tags]
        total = 0.0
        stream = hist_morphs + cur
        start = len(hist_morphs)
        for i in range(start, len(stream)):
            ctx = stream[max(0, i - (lm.order - 1)): i]
            total += lm.logprob(stream[i], ctx)
        return total

    return score
