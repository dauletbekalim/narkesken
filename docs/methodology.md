# Methodology

This note documents the technical design of Narkesken in more depth than the
README: why each component exists, what algorithm it implements, and where the
known limitations are.

## 1. Why Kazakh ASR is hard

Kazakh is a **Kipchak Turkic**, **agglutinative** language. Two properties
dominate the engineering:

1. **Productive suffixation.** A nominal root combines a plural slot, a
   possessive slot, and a case slot; a verbal root combines voice, negation,
   tense/aspect, and person. Each slot has several *allomorphs* selected by
   phonology. The number of distinct *surface* word forms per root is therefore
   large, and the long tail of inflected forms is heavy. A word-level model —
   whether a language model or a fixed-vocabulary acoustic decoder — sees an
   enormous type inventory and a correspondingly severe OOV problem.

2. **Low resource.** Transcribed Kazakh speech is scarce relative to English or
   Russian, and most off-the-shelf speech models were trained on those higher
   resource languages. In a Kazakh-speaking (frequently Kazakh/Russian
   bilingual) setting they mishear both the phonology and the morphology.

Narkesken's design choices all follow from these two facts.

## 2. Acoustic model: wav2vec2 + character CTC

We fine-tune a multilingual **wav2vec2-XLSR** checkpoint with a **character-level
CTC** head over the Kazakh Cyrillic alphabet.

- *Why XLSR:* self-supervised multilingual pretraining gives a phonetically
  informed starting point that transfers to a low-resource target far better
  than an English-only model or a from-scratch model.
- *Why character CTC (not a word or subword output):* a character head is not
  committed to any fixed word vocabulary, so at inference it can spell out
  inflected forms it never saw as whole words in training — exactly the property
  an agglutinative language needs. CTC's monotonic alignment is a good fit for
  speech and keeps the decoder simple.

Regularization matters more than usual in the low-resource regime: we freeze the
convolutional feature encoder, use dropout on attention/hidden/projection
layers, enable wav2vec2's built-in time masking, and add an external
**SpecAugment** + waveform-augmentation pipeline (`narkesken.data.augment`).

## 3. Morphological analysis (`narkesken.morphology`)

A lexicon-free, finite-state-style analyzer that decomposes a surface word into
a root and an ordered morpheme chain. It has two ingredients:

- **Phonology** (`phonology.py`): vowel-harmony classification (front/back, plus
  a weak labial harmony for high vowels) and final-phoneme classification
  (vowel / voiced / voiceless / sonorant) for consonant assimilation.
- **Paradigms** (`suffixes.py`): each grammatical slot lists its allomorphs,
  each annotated with the phonological context that licenses it.

The analyzer (`segmenter.py`) runs a small **beam search** that peels suffixes
from the end of the word in reverse canonical order, over both the nominal chain
(PLURAL → POSSESSIVE → CASE) and the verbal chain (NEGATION → TENSE → PERSON).
Each candidate parse is scored by summing an allomorph-licensing reward
(harmony + assimilation agreement — the signal a flat suffix list cannot
provide), minus an over-segmentation penalty and a short/vowelless-root penalty.

**Known limitation.** Because there is no root lexicon, the analyzer cannot rule
out a phonologically valid but lexically nonsensical parse (e.g. reading a noun
as a verb + personal ending). A production system would compose this with a
lexical FST such as `apertium-kaz`. The scoring is a heuristic approximation,
not a validated linguistic model.

## 4. Decoding (`narkesken.decoding`)

- **CTC prefix beam search** (`beam_search.py`), after Hannun et al. (2014):
  maintains, per prefix, the blank/non-blank ending probabilities and extends
  frame-by-frame under the CTC collapse rule.
- **Shallow fusion**: a language-model factor `α·log p_LM + β·|words|` is added
  each time a word delimiter completes a word (β is a word-insertion bonus that
  offsets the LM's length bias).
- **Modified Kneser-Ney n-gram LM** (`ngram_lm.py`), after Chen & Goodman
  (1998): interpolated, with the three discounts `D₁,D₂,D₃₊` estimated from the
  counts-of-counts and continuation counts for the lower orders.
- **Morpheme factoring** (`morpheme_lm_scorer`): the LM is trained on *morpheme*
  sequences produced by the analyzer, so a sparse inflected word is scored as a
  sequence of pieces the LM has actually observed. This is the standard remedy
  for the word-level OOV explosion in Turkic ASR.

## 5. Evaluation (`narkesken.metrics`)

- **WER / CER** with a self-contained Levenshtein aligner (so the alignment,
  not just the score, is available downstream).
- **Morphology-aware WER (M-WER)** (`morph_wer.py`): reuses the word-level
  alignment, but replaces the unit substitution cost with a morphologically
  informed one. For an aligned substitution, both words are analyzed; if the
  **roots match**, the cost is `w_suffix ·` (normalized suffix-chain edit
  distance) — a *small* charge for an inflection slip; if the **roots differ**,
  the cost is at least `w_root` — a *large* charge for a lexical error.

  The intuition: a wrong case/tense/possession marker usually means the model
  *understood the word and mis-inflected it*, whereas a wrong root means it
  *did not recognize the word at all*. Plain WER conflates the two; M-WER
  separates them and also returns a breakdown (exact / suffix-only / root /
  insertion / deletion) for error analysis.

  Setting `w_suffix = w_root = 1` recovers (approximately) ordinary WER; setting
  `w_suffix = 0` gives a fully inflection-insensitive metric. The default
  `w_root = 1.0, w_suffix = 0.35` sits deliberately in between.

## 6. What would come next

- Compose the analyzer with a real lexical/morphological FST to remove
  nonsensical parses.
- Replace the per-digit number normalizer with a full number-to-words model.
- Human-anchor the M-WER weights against listener judgments of severity, so
  `w_root` / `w_suffix` reflect measured downstream cost rather than a prior.
- Add a second-pass neural LM rescorer on top of the n-gram first pass.
