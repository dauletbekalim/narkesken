"""Kazakh (Cyrillic) text normalization for ASR transcripts.

Maps raw transcripts onto the compact, consistent character set that a
character-level CTC head predicts. The pipeline is intentionally conservative
and fully reversible only up to case/punctuation: it (1) NFC-normalizes Unicode,
(2) lowercases, (3) repairs common Latin/Russian homoglyph contamination,
(4) expands a small set of digits/abbreviations into spoken Kazakh words,
(5) strips punctuation to word boundaries, and (6) collapses whitespace.

It deliberately does **not** do morphological analysis — see
:mod:`narkesken.morphology` for that.
"""

from __future__ import annotations

import re
import sys
import unicodedata

# The 42-letter Kazakh Cyrillic alphabet (lowercase) plus the CTC word-delimiter
# space. Anything outside this set is turned into a boundary by default.
KAZAKH_LOWER = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
ALLOWED_CHARS = set(KAZAKH_LOWER) | {" "}

# Latin homoglyphs that leak in from mixed-keyboard input; mapped to their
# Cyrillic look-alikes. Applied only to isolated Latin letters (see _fix_homoglyphs).
LATIN_TO_CYRILLIC = {
    "a": "а", "e": "е", "o": "о", "c": "с", "p": "р",
    "x": "х", "y": "у", "k": "к", "m": "м", "t": "т",
    "h": "һ", "b": "в",
}

# Spoken-form expansion for the ten digits (Kazakh cardinal numerals).
DIGIT_WORDS = {
    "0": "нөл", "1": "бір", "2": "екі", "3": "үш", "4": "төрт",
    "5": "бес", "6": "алты", "7": "жеті", "8": "сегіз", "9": "тоғыз",
}

# A few high-frequency abbreviations worth expanding before stripping periods.
ABBREVIATIONS = {
    "т.б.": "тағы басқа",
    "т.с.с.": "тағы сол сияқты",
    "және т.б.": "және тағы басқа",
    "ғ.": "ғасыр",
    "жыл.": "жыл",
}

_WHITESPACE_RE = re.compile(r"\s+")
_LATIN_RUN_RE = re.compile(r"[a-z]+")


def _fix_homoglyphs(text: str) -> str:
    """Convert stray Latin letters to Cyrillic look-alikes.

    Only rewrites Latin runs that are *fully* mappable (i.e. every letter has a
    Cyrillic homoglyph). Genuine English words (with unmappable letters) are
    left for the punctuation/whitelist stage to strip, so we don't silently
    mangle e.g. proper nouns into gibberish.
    """
    def repl(m: re.Match) -> str:
        run = m.group(0)
        if all(ch in LATIN_TO_CYRILLIC for ch in run):
            return "".join(LATIN_TO_CYRILLIC[ch] for ch in run)
        return run

    return _LATIN_RUN_RE.sub(repl, text)


def _expand_digits(text: str) -> str:
    """Read out standalone digit sequences as space-separated Kazakh numerals.

    This is a simple per-digit reading (e.g. ``12`` -> ``бір екі``), which is a
    reasonable stand-in for a full number-to-words normalizer and keeps every
    token in-vocabulary.
    """
    return re.sub(
        r"\d+",
        lambda m: " ".join(DIGIT_WORDS[d] for d in m.group(0)),
        text,
    )


def normalize_text(text: str | None, *, expand_digits: bool = True,
                   keep_unknown: bool = False) -> str:
    """Normalize a Kazakh transcript to the model's character set."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text).lower()

    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, f" {full} ")

    text = _fix_homoglyphs(text)

    if expand_digits:
        text = _expand_digits(text)

    # Replace anything outside the alphabet with a boundary space (unless asked
    # to keep unknowns for debugging).
    cleaned = []
    for ch in text:
        if ch in ALLOWED_CHARS:
            cleaned.append(ch)
        elif keep_unknown:
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    text = "".join(cleaned)

    return _WHITESPACE_RE.sub(" ", text).strip()


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "Сәлеметсіз бе, әлем! 12 saǵat, т.б."
    print(f"raw:        {sample!r}")
    print(f"normalized: {normalize_text(sample)!r}")
