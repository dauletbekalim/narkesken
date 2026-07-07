"""Kazakh morphological analysis for morphology-aware ASR evaluation."""

from .segmenter import Analysis, KazakhSegmenter, Morpheme, root_suffix, segment

__all__ = [
    "Analysis",
    "KazakhSegmenter",
    "Morpheme",
    "segment",
    "root_suffix",
]
