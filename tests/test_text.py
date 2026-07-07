import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.text import build_vocab, normalize_text


def test_normalize_strips_punct_and_lowercases():
    assert normalize_text("Сәлеметсіз бе, әлем!") == "сәлеметсіз бе әлем"


def test_normalize_expands_digits():
    assert normalize_text("12", expand_digits=True) == "бір екі"


def test_normalize_collapses_whitespace():
    assert normalize_text("  көп   бос  орын ") == "көп бос орын"


def test_build_vocab_has_special_tokens():
    vocab = build_vocab(["сәлем әлем"])
    for tok in ("|", "[UNK]", "[PAD]"):
        assert tok in vocab
    # indices are unique and contiguous
    assert sorted(vocab.values()) == list(range(len(vocab)))
