"""Kazakh (Cyrillic) text normalization for ASR transcripts.

The goal is to map raw transcripts onto the small, consistent character set that
a character-level CTC head will predict. This is intentionally conservative:
it lowercases, keeps the Kazakh Cyrillic alphabet, strips punctuation, and
collapses whitespace. It does NOT do morphological analysis (see the
morphology-aware metric stub in ``evaluate.py`` for that idea).

Run standalone to normalize a line of text:

    python data_prep/normalize.py "Сәлеметсіз бе, әлем!"
"""

from __future__ import annotations

import re
import sys
import unicodedata

# The 42-letter Kazakh Cyrillic alphabet (lowercase), plus the space that CTC
# uses as a word delimiter. Anything outside this set is dropped by default.
KAZAKH_LOWER = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
ALLOWED_CHARS = set(KAZAKH_LOWER) | {" "}

# A few common non-standard substitutions seen in real-world Kazakh text.
# (Some keyboards/encodings emit Russian look-alikes for Kazakh-specific letters,
# or use the Latin apostrophe forms.) Extend this map as your data requires.
CHAR_SUBSTITUTIONS = {
    "ә": "ә",  # already ә, but normalizes any decomposed form
    "h": "һ",
    # Latin homoglyphs occasionally leak into otherwise-Cyrillic text:
    "a": "а",
    "e": "е",
    "o": "о",
    "c": "с",
    "p": "р",
    "x": "х",
    "y": "у",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, *, keep_unknown: bool = False) -> str:
    """Normalize a Kazakh transcript to the model's character set.

    Args:
        text: Raw transcript string.
        keep_unknown: If True, characters outside the Kazakh alphabet are kept
            (useful for debugging what is being dropped). Default False.

    Returns:
        A lowercased, punctuation-free string over the Kazakh alphabet + spaces.
    """
    if text is None:
        return ""

    # Normalize Unicode so composed/decomposed forms compare equal, then lowercase.
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    # Apply per-character substitutions before filtering.
    text = "".join(CHAR_SUBSTITUTIONS.get(ch, ch) for ch in text)

    # Replace anything that is not a letter/space with a space, so that
    # punctuation acts as a word boundary rather than gluing words together.
    cleaned_chars = []
    for ch in text:
        if ch in ALLOWED_CHARS:
            cleaned_chars.append(ch)
        elif keep_unknown:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)

    # Collapse runs of whitespace and trim.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_vocab(transcripts: list[str]) -> dict[str, int]:
    """Build a character->index vocabulary from normalized transcripts.

    Adds the CTC-standard special tokens: ``|`` for the word delimiter
    (the space is remapped to ``|`` as is conventional for wav2vec2 CTC),
    plus ``[UNK]`` and ``[PAD]``.
    """
    chars: set[str] = set()
    for t in transcripts:
        chars.update(normalize_text(t))
    chars.discard(" ")

    vocab = {ch: i for i, ch in enumerate(sorted(chars))}
    vocab["|"] = len(vocab)  # word delimiter (replaces space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "Сәлеметсіз бе, әлем! 123"
    print(f"raw:        {sample!r}")
    print(f"normalized: {normalize_text(sample)!r}")
