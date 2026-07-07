"""Narkesken — Kazakh-language speech recognition.

A wav2vec2-based ASR toolkit for Kazakh (a low-resource, agglutinative Turkic
language), with first-class support for the two things that make Kazakh ASR hard:

* **Morphology** (:mod:`narkesken.morphology`) — a rule-based analyzer with vowel
  harmony and consonant assimilation, used both for LM factoring and for
  morphology-aware evaluation.
* **Decoding** (:mod:`narkesken.decoding`) — CTC prefix beam search with modified
  Kneser-Ney n-gram LM shallow fusion, morpheme-factored to beat OOV.
* **Metrics** (:mod:`narkesken.metrics`) — WER/CER plus M-WER, which weights root
  errors and suffix-chain errors differently.

The heavy DL stack (torch/transformers/datasets) is imported lazily, so the
pure-Python components can be used on their own.
"""

from .__version__ import __version__

__all__ = ["__version__"]
