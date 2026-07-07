import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.metrics import MorphologyAwareWER, align, word_error_rate
from narkesken.metrics.wer import Op


def test_alignment_basic():
    ops = align("a b c".split(), "a x c".split())
    kinds = [o.op for o in ops]
    assert kinds == [Op.MATCH, Op.SUB, Op.MATCH]


def test_wer_counts_full_word_errors():
    refs = ["мен мектепке барамын"]
    hyps = ["мен мектепте барамын"]  # one word differs
    assert abs(word_error_rate(refs, hyps) - 1 / 3) < 1e-9


def test_mwer_discounts_suffix_error():
    refs = ["мен мектепке барамын"]
    hyps = ["мен мектепте барамын"]  # same root мектеп, wrong case suffix
    m = MorphologyAwareWER(w_root=1.0, w_suffix=0.35).compute(refs, hyps)
    # A suffix-only error must be cheaper than a full unit error.
    assert m.suffix_only == 1
    assert m.root_errors == 0
    assert m.m_wer < m.wer_equivalent


def test_mwer_full_penalty_for_root_error():
    refs = ["ол үйге келді"]
    hyps = ["ол үйге кетті"]  # келді vs кетті -> different root
    m = MorphologyAwareWER().compute(refs, hyps)
    assert m.root_errors == 1
    assert m.suffix_only == 0


def test_mwer_equals_wer_when_suffix_weight_high():
    refs = ["мен мектепке барамын"]
    hyps = ["мен мектепте барамын"]
    m = MorphologyAwareWER(w_root=1.0, w_suffix=1.0).compute(refs, hyps)
    # With full suffix weight the suffix error approaches a unit error.
    assert m.m_wer <= m.wer_equivalent + 1e-9
