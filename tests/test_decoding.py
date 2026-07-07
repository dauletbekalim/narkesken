import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.decoding import ModifiedKneserNeyLM, ctc_prefix_beam_search
from narkesken.decoding.beam_search import greedy_ctc_decode


# id map: 0=blank, 1='а', 2='б', 3='|'(space)
ID_TO_CHAR = {0: "[PAD]", 1: "а", 2: "б", 3: "|"}
BLANK = 0


def _one_hot_logprobs(seq_ids, vocab_size=4, hot=0.9):
    rows = []
    for i in seq_ids:
        row = [math.log((1 - hot) / (vocab_size - 1))] * vocab_size
        row[i] = math.log(hot)
        rows.append(row)
    return rows


def test_greedy_collapses_repeats_and_blanks():
    # 'а','а'(repeat, collapsed),blank,'а' -> "аа" ; then space, 'б'
    frames = _one_hot_logprobs([1, 1, 0, 1, 3, 2])
    out = greedy_ctc_decode(frames, ID_TO_CHAR, blank_id=BLANK)
    assert out == "аа б"


def test_beam_search_recovers_simple_sequence():
    frames = _one_hot_logprobs([1, 0, 2, 0, 1])  # а б а
    results = ctc_prefix_beam_search(frames, ID_TO_CHAR, blank_id=BLANK, beam_width=8)
    assert results
    assert results[0][0] == "аба"


def test_kneser_ney_probabilities_sum_reasonably():
    corpus = [
        ["мен", "мектепке", "барамын"],
        ["мен", "үйге", "барамын"],
        ["ол", "мектепке", "келді"],
    ]
    lm = ModifiedKneserNeyLM.train(corpus, order=2)
    p = lm.prob("барамын", ["мектепке"])
    assert 0.0 <= p <= 1.0
    # A seen continuation should beat an unseen one.
    assert lm.prob("барамын", ["мектепке"]) > lm.prob("келді", ["үйге"])


def test_lm_save_load_roundtrip(tmp_path):
    corpus = [["а", "б", "а"], ["а", "а", "б"]]
    lm = ModifiedKneserNeyLM.train(corpus, order=2)
    path = tmp_path / "toy.lm"
    lm.save(path)
    lm2 = ModifiedKneserNeyLM.load(path)
    assert lm2.order == lm.order
    assert abs(lm2.prob("б", ["а"]) - lm.prob("б", ["а"])) < 1e-9
