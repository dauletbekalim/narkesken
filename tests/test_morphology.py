import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narkesken.morphology import KazakhSegmenter, root_suffix
from narkesken.morphology.phonology import final_phoneme_class, harmony_class


def test_harmony_class():
    assert harmony_class("мектеп") == "front"   # last vowel е (front)
    assert harmony_class("бала") == "back"       # last vowel а (back)
    assert harmony_class("үй") == "front"


def test_final_phoneme_class():
    assert final_phoneme_class("бала") == "vowel"
    assert final_phoneme_class("мектеп") == "voiceless"  # п
    assert final_phoneme_class("үй") == "sonorant"       # й


def test_segments_plural_root():
    root, sufs = root_suffix("мектептер")
    assert root == "мектеп"
    assert sufs and sufs[0].startswith("тер") or "тер" in sufs


def test_segments_case_on_vowel_stem():
    seg = KazakhSegmenter()
    a = seg.best("балаларына")
    assert a.root == "бала"
    assert "PL" in a.tags


def test_bare_root_stays_whole():
    root, sufs = root_suffix("үй")
    assert root == "үй"
    assert sufs == []


def test_verbal_negation_chain():
    seg = KazakhSegmenter()
    a = seg.best("келмедім")
    # Should recover the verb root 'кел' and include a NEG marker.
    assert a.root == "кел"
    assert "NEG" in a.tags
