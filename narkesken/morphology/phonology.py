"""Kazakh phonology: vowel harmony and consonant-assimilation classes.

These tables drive allomorph selection in :mod:`narkesken.morphology.segmenter`.
Kazakh suffixation is not concatenative string-gluing — each suffix has several
surface *allomorphs*, and which one appears is determined by two phonological
systems:

1. **Vowel harmony.** Kazakh vowels split into a *front* (палатальный) and a
   *back* (велярный) set. A suffix vowel must agree with the harmony class of
   the last vowel in the stem. (Kazakh has, additionally, a weak labial /
   rounding harmony that we model only for the high vowels ұ/ү/ы/і.)

2. **Consonant assimilation.** The initial consonant of many suffixes alternates
   between a voiced stop, a voiceless stop, and a nasal/liquid depending on the
   final phoneme of the stem (vowel vs. voiced vs. voiceless vs. nasal/sonorant).

References
----------
- Vinogradov & Kirchner, morphophonology of Turkic vowel harmony.
- Washington et al. (2014), "Finite-state morphological transducers for three
  Kypchak languages" — the allomorph-conditioning logic here follows the same
  two-axis (harmony × assimilation) design used in Apertium's ``apertium-kaz``.
"""

from __future__ import annotations

# --- Vowel inventory --------------------------------------------------------
FRONT_VOWELS = set("әеіөүэ")          # front / palatal
BACK_VOWELS = set("аоұыя")            # back / velar (я treated as back-ish 'a')
# 'и' and 'у' are historically diphthongal/neutral; we resolve them by the
# surrounding vowel and default to *front* when ambiguous (see harmony_class).
NEUTRAL_VOWELS = set("иуё")
VOWELS = FRONT_VOWELS | BACK_VOWELS | NEUTRAL_VOWELS

ROUNDED_VOWELS = set("оөұүёю")        # for weak labial harmony of high vowels
HIGH_VOWELS = set("ыіұүи")

# --- Consonant classes (for suffix-initial assimilation) --------------------
# Voiceless obstruents: a suffix starting with a stop surfaces as its voiceless
# allomorph (т-, қ-, п-, с-) after these.
VOICELESS = set("пфктсшщхһцч")
# Voiced obstruents + everything else that triggers the voiced allomorph (д-, ғ-).
VOICED = set("бвгғджзһ")
# Sonorants / nasals: trigger the л-/н- (liquid/nasal) allomorphs.
SONORANTS = set("рлмнңйуview".replace("view", ""))  # р л м н ң й (у as glide)
SONORANTS = set("рлмнңй")


def last_vowel(word: str) -> str | None:
    """Return the last vowel character in ``word`` (or ``None``)."""
    for ch in reversed(word):
        if ch in VOWELS:
            return ch
    return None


def harmony_class(word: str) -> str:
    """Classify a stem as ``"front"`` or ``"back"`` for vowel harmony.

    Uses the last non-neutral vowel. Falls back to ``"back"`` (the statistically
    dominant class for consonant-final loanwords) when a stem contains only
    neutral vowels or no vowel at all.
    """
    v = None
    for ch in reversed(word):
        if ch in FRONT_VOWELS:
            return "front"
        if ch in BACK_VOWELS:
            return "back"
        if ch in NEUTRAL_VOWELS and v is None:
            v = ch  # remember but keep looking for a decisive vowel
    if v in {"и", "ё"}:  # и/ё lean front in native phonotactics
        return "front"
    return "back"


def is_rounded(word: str) -> bool:
    """True if the last vowel is rounded (for labial harmony of high vowels)."""
    v = last_vowel(word)
    return v in ROUNDED_VOWELS


def final_phoneme_class(word: str) -> str:
    """Classify the final phoneme of a stem for consonant assimilation.

    Returns one of ``"vowel"``, ``"voiced"``, ``"voiceless"``, ``"sonorant"``.
    """
    if not word:
        return "vowel"
    last = word[-1]
    if last in VOWELS:
        return "vowel"
    if last in VOICELESS:
        return "voiceless"
    if last in SONORANTS:
        return "sonorant"
    if last in VOICED:
        return "voiced"
    # Unknown consonant (e.g. from a loanword) -> behave like a voiced obstruent.
    return "voiced"
