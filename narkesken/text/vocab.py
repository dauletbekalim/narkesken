"""Character vocabulary construction for the wav2vec2 CTC head."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .normalize import normalize_text

WORD_DELIMITER = "|"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


def build_vocab(transcripts: list[str], *, min_count: int = 1) -> dict[str, int]:
    """Build a character->index vocab from (already collectible) transcripts.

    Characters occurring fewer than ``min_count`` times are dropped (they will
    map to ``[UNK]``), which guards against rare OCR/encoding noise inflating
    the softmax. The space is remapped to the CTC word-delimiter ``|``.
    """
    counts: Counter[str] = Counter()
    for t in transcripts:
        counts.update(normalize_text(t).replace(" ", ""))

    chars = sorted(ch for ch, c in counts.items() if c >= min_count)
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab[WORD_DELIMITER] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    vocab[PAD_TOKEN] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
