"""Kazakh nominal & verbal suffix paradigms with allomorph conditioning.

Each :class:`Paradigm` describes one grammatical slot (e.g. PLURAL, the dative
CASE, a PERSON marker) and enumerates its surface *allomorphs* together with the
phonological context that licenses each one. The morphological analyzer in
:mod:`narkesken.morphology.segmenter` consults these to (a) recognize a suffix
in a surface word and (b) check that the allomorph it found is phonologically
consistent with the preceding stem — a consistency signal that a naive
suffix-list stripper cannot provide.

Order of the ``ORDER`` list encodes the canonical Kazakh morpheme order:

    root (+ derivation) + PLURAL + POSSESSIVE + CASE          (nominal)
    root (+ VOICE/NEG) + TENSE/ASPECT + PERSON                (verbal)

The inventory is intentionally broad but not exhaustive; it targets the
high-frequency inflectional morphology that dominates running speech.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .phonology import final_phoneme_class, harmony_class, is_rounded

# A phonological context predicate: given the *stem so far*, is this allomorph
# licensed? Predicates are cheap booleans over harmony/assimilation classes.
Context = "callable"  # documentation alias


@dataclass(frozen=True)
class Allomorph:
    surface: str            # the surface string, e.g. "тер"
    harmony: str | None = None      # "front" | "back" | None (any)
    after: tuple[str, ...] = ()     # licensing final-phoneme classes; () = any
    rounded: bool | None = None     # labial-harmony requirement, if any

    def licensed(self, stem: str) -> bool:
        if self.harmony is not None and harmony_class(stem) != self.harmony:
            return False
        if self.after and final_phoneme_class(stem) not in self.after:
            return False
        if self.rounded is not None and is_rounded(stem) != self.rounded:
            return False
        return True


@dataclass(frozen=True)
class Paradigm:
    slot: str               # grammatical slot, e.g. "PLURAL"
    tag: str                # short morphosyntactic tag, e.g. "PL"
    allomorphs: tuple[Allomorph, ...]
    optional: bool = True   # inflectional slots are generally optional

    def match_at_end(self, word: str) -> list[tuple[str, str, bool]]:
        """Return candidate splits of ``word`` ending in this paradigm.

        Yields ``(stem, surface_suffix, licensed)`` for every allomorph that is
        a suffix of ``word`` (longest first), where ``licensed`` reports whether
        the allomorph is phonologically consistent with ``stem``.
        """
        out: list[tuple[str, str, bool]] = []
        for allo in sorted(self.allomorphs, key=lambda a: len(a.surface), reverse=True):
            s = allo.surface
            if len(word) - len(s) >= 1 and word.endswith(s):
                stem = word[: -len(s)]
                out.append((stem, s, allo.licensed(stem)))
        return out


def _harmony_pair(front: str, back: str, **kw) -> tuple[Allomorph, ...]:
    return (
        Allomorph(front, harmony="front", **kw),
        Allomorph(back, harmony="back", **kw),
    )


# --- Nominal paradigms ------------------------------------------------------

PLURAL = Paradigm(
    "PLURAL", "PL",
    (
        # л-allomorphs after vowels & sonorants (except nasals ң/м where н-set wins)
        *_harmony_pair("лер", "лар", after=("vowel", "sonorant")),
        # д-allomorphs after voiced obstruents
        *_harmony_pair("дер", "дар", after=("voiced",)),
        # т-allomorphs after voiceless obstruents
        *_harmony_pair("тер", "тар", after=("voiceless",)),
    ),
)

POSSESSIVE = Paradigm(
    "POSSESSIVE", "POSS",
    (
        # 1sg -(ы)м/-(і)м, 2sg -(ы)ң/-(і)ң, 3 -(с)ы/-(с)і, 1pl -(ы)мыз/-(і)міз ...
        Allomorph("ым", harmony="back", after=("voiced", "voiceless", "sonorant")),
        Allomorph("ім", harmony="front", after=("voiced", "voiceless", "sonorant")),
        Allomorph("м", after=("vowel",)),
        Allomorph("ың", harmony="back", after=("voiced", "voiceless", "sonorant")),
        Allomorph("ің", harmony="front", after=("voiced", "voiceless", "sonorant")),
        Allomorph("ң", after=("vowel",)),
        Allomorph("сы", harmony="back", after=("vowel",)),
        Allomorph("сі", harmony="front", after=("vowel",)),
        Allomorph("ы", harmony="back", after=("voiced", "voiceless", "sonorant")),
        Allomorph("і", harmony="front", after=("voiced", "voiceless", "sonorant")),
        Allomorph("мыз", harmony="back"),
        Allomorph("міз", harmony="front"),
        Allomorph("ңыз", harmony="back"),
        Allomorph("ңіз", harmony="front"),
    ),
)

CASE = Paradigm(
    "CASE", "CASE",
    (
        # Genitive -ның/-нің/-дың/-дің/-тың/-тің
        *_harmony_pair("нің", "ның", after=("vowel", "sonorant")),
        *_harmony_pair("дің", "дың", after=("voiced",)),
        *_harmony_pair("тің", "тың", after=("voiceless",)),
        # Dative -ға/-ге/-қа/-ке (velar/uvular alternation with harmony)
        Allomorph("ге", harmony="front", after=("vowel", "voiced", "sonorant")),
        Allomorph("ға", harmony="back", after=("vowel", "voiced", "sonorant")),
        Allomorph("ке", harmony="front", after=("voiceless",)),
        Allomorph("қа", harmony="back", after=("voiceless",)),
        # Dative after a 3rd-person possessive vowel takes the epenthetic -н-:
        # бала-сы-на, кітап-ы-на. Licensed only on a vowel-final (possessive) stem.
        Allomorph("на", harmony="back", after=("vowel",)),
        Allomorph("не", harmony="front", after=("vowel",)),
        # Accusative -ны/-ні/-ды/-ді/-ты/-ті
        *_harmony_pair("ні", "ны", after=("vowel", "sonorant")),
        *_harmony_pair("ді", "ды", after=("voiced",)),
        *_harmony_pair("ті", "ты", after=("voiceless",)),
        # Locative -да/-де/-та/-те
        *_harmony_pair("де", "да", after=("vowel", "voiced", "sonorant")),
        *_harmony_pair("те", "та", after=("voiceless",)),
        # Ablative -дан/-ден/-тан/-тен/-нан/-нен
        *_harmony_pair("нен", "нан", after=("sonorant",)),
        *_harmony_pair("ден", "дан", after=("vowel", "voiced")),
        *_harmony_pair("тен", "тан", after=("voiceless",)),
        # Instrumental -мен/-бен/-пен
        Allomorph("мен", after=("vowel", "sonorant")),
        Allomorph("бен", after=("voiced",)),
        Allomorph("пен", after=("voiceless",)),
    ),
)

# --- Verbal paradigms (high-frequency subset) -------------------------------

NEGATION = Paradigm(
    "NEGATION", "NEG",
    (
        *_harmony_pair("ме", "ма", after=("vowel", "voiced", "sonorant")),
        *_harmony_pair("бе", "ба", after=("voiced",)),
        *_harmony_pair("пе", "па", after=("voiceless",)),
    ),
)

TENSE = Paradigm(
    "TENSE", "TAM",
    (
        # Past -ды/-ді/-ты/-ті
        *_harmony_pair("ді", "ды", after=("vowel", "voiced", "sonorant")),
        *_harmony_pair("ті", "ты", after=("voiceless",)),
        # Perfective participle -ған/-ген/-қан/-кен
        Allomorph("ген", harmony="front", after=("vowel", "voiced", "sonorant")),
        Allomorph("ған", harmony="back", after=("vowel", "voiced", "sonorant")),
        Allomorph("кен", harmony="front", after=("voiceless",)),
        Allomorph("қан", harmony="back", after=("voiceless",)),
        # Present/aorist -а/-е/-й
        Allomorph("й", after=("vowel",)),
        Allomorph("е", harmony="front", after=("voiced", "voiceless", "sonorant")),
        Allomorph("а", harmony="back", after=("voiced", "voiceless", "sonorant")),
    ),
)

PERSON = Paradigm(
    "PERSON", "PERS",
    (
        # Present/future personal endings (attach to the aorist vowel).
        Allomorph("мын", harmony="back"),
        Allomorph("мін", harmony="front"),
        Allomorph("сың", harmony="back"),
        Allomorph("сің", harmony="front"),
        Allomorph("сыз", harmony="back"),
        Allomorph("сіз", harmony="front"),
        Allomorph("мыз", harmony="back"),
        Allomorph("міз", harmony="front"),
        # Past-tense personal endings attach to the past marker -ды/-ді, whose
        # final vowel licenses these bare-consonant endings (келді-м, оқыды-қ).
        # Restricting them to vowel-final stems keeps them off consonant-final
        # nouns, where a stray "м"/"ң" would be a false segmentation.
        Allomorph("м", after=("vowel",)),          # 1sg past
        Allomorph("ң", after=("vowel",)),          # 2sg past
        Allomorph("қ", harmony="back", after=("vowel",)),   # 1pl past
        Allomorph("к", harmony="front", after=("vowel",)),  # 1pl past
    ),
)

# Canonical suffix order, from the root outward. The analyzer peels from the
# END of the word, i.e. it walks this list in reverse.
ORDER: tuple[Paradigm, ...] = (
    # nominal chain
    PLURAL, POSSESSIVE, CASE,
    # verbal chain
    NEGATION, TENSE, PERSON,
)

# Fast lookup used by the analyzer.
NOMINAL_CHAIN = (PLURAL, POSSESSIVE, CASE)
VERBAL_CHAIN = (NEGATION, TENSE, PERSON)
