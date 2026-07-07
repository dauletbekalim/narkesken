# narkesken — kazakh speech recognition

**a wav2vec2-based automatic speech recognition (ASR) toolkit for Kazakh, a
low-resource, agglutinative Turkic language — with morphology-aware decoding and
evaluation built in.**

narkesken pairs a standard XLSR fine-tuning pipeline with two things most ASR
stacks lack for a language like Kazakh: a **rule-based morphological analyzer**
(vowel harmony + consonant assimilation) and a **morphology-aware error metric**
that scores a wrong grammatical suffix differently from a wrong root. the
morphology, metrics, language-model, and decoding modules are pure Python and
run with no deep-learning dependencies at all; the acoustic training/inference
path uses HuggingFace `transformers` + `torch`.

> **status.** actively developed research code. the pure-Python components
> (morphology, metrics, n-gram LM, CTC decoder, text normalization) are covered
> by the test suite and runnable today via `scripts/demo_morphology.py`. the
> acoustic training path is a complete, standard wav2vec2-CTC pipeline intended
> to be pointed at a real Kazakh corpus (none ships with the repo — see
> [data](#data)). no benchmark numbers are quoted here on purpose; see
> [a note on numbers](#a-note-on-numbers).

---

## background

narkesken began as a Kazakh speech-recognition project **piloted in 2024–2025 in
rural Kazakh-speaking schools**, with **~170 active users** and support from a
**small government grant**. this repository is the clean, documented reference
implementation of that approach — the wav2vec2 fine-tuning recipe plus the
morphology-aware tooling that the agglutinative structure of Kazakh demands.

## why kazakh ASR is a distinct problem

Kazakh is **agglutinative**: a single root takes an ordered chain of suffixes —
for nouns `root (+PLURAL)(+POSSESSIVE)(+CASE)`, for verbs
`root (+NEGATION)(+TENSE/ASPECT)(+PERSON)` — and each suffix has several surface
**allomorphs** chosen by vowel harmony and consonant assimilation. so:

- the number of distinct *surface* word forms per root is large, and rare
  inflected forms are everywhere in ordinary speech;
- a word-level language model or fixed-vocabulary decoder faces a severe
  out-of-vocabulary (OOV) problem;
- and because Kazakh is **low-resource**, most available speech models are
  trained on Russian or English and mishear both its phonology and its
  morphology.

every design choice below follows from these facts. the long version is in
[`docs/methodology.md`](docs/methodology.md).

## approach at a glance

| stage | module | what it does |
|-------|--------|--------------|
| acoustic model | `narkesken.training` | fine-tunes **wav2vec2-XLSR** with a **character-level CTC** head over the Kazakh Cyrillic alphabet. character output means the model can spell inflected forms it never saw as whole words. |
| morphology | `narkesken.morphology` | lexicon-free analyzer: **vowel harmony** + **consonant assimilation** drive allomorph selection; a beam search peels the nominal/verbal suffix chains and scores parses by phonological licensing. |
| language model | `narkesken.decoding.ngram_lm` | **modified Kneser-Ney** n-gram LM (Chen & Goodman 1998), trainable over **morphemes** to sidestep word-level OOV. |
| decoding | `narkesken.decoding.beam_search` | **CTC prefix beam search** (Hannun et al. 2014) with **n-gram LM shallow fusion**, optionally morpheme-factored. |
| evaluation | `narkesken.metrics` | WER/CER **plus M-WER**, a morphology-aware error rate that weights root vs. suffix-chain errors differently. |
| data | `narkesken.data` | dataset packaging, Kazakh text normalization, SpecAugment + waveform augmentation, and a synthetic sample-data generator. |

## the headline idea: morphology-aware error rate (M-WER)

standard **word error rate** counts every non-matching word as one full error.
for Kazakh that is badly miscalibrated. consider a reference `мектепке`
(“to school”, dative) and a hypothesis `мектепте` (“at school”, locative):

- **WER**: one full word error — the same penalty as an unrelated word.
- **reality**: the model got the root `мектеп` right and slipped one case
  marker. it *understood the word* and mis-inflected it.

a wrong **root** (`келді` → `кетті`, “came” → “left”) is a fundamentally worse
failure than a wrong **suffix**, yet WER can't tell them apart — so a single WER
number hides whether a model's errors are "heard the word, botched the grammar"
or "didn't recognize the word." those have very different downstream cost (e.g.
for a reading tutor used by schoolchildren).

**M-WER** (`narkesken.metrics.morph_wer`) fixes this: it keeps the standard
Levenshtein alignment, but for each substituted word it analyzes both words and
charges `w_suffix ·` (suffix-chain edit distance) when the roots match, versus
`w_root` when they differ. it also returns a full breakdown — exact /
suffix-only / root / insertion / deletion — for error analysis.

```
$ python scripts/demo_morphology.py
...
  WER   = 0.2222   (мектепте and кетті each count as 1 full error)
  M-WER = 0.1889
    exact=7 suffix-only=1 root-err=1 ins=0 del=0
    suffix-only cases the metric discounted:
      - 'мектепке' -> 'мектепте'  (root ok, suffix Δ, cost=0.35)
```

## repository layout

```
narkesken/
├── narkesken/                     # the installable package (pure-Python core)
│   ├── morphology/                # phonology, suffix paradigms, the analyzer
│   │   ├── phonology.py           #   vowel harmony + consonant assimilation
│   │   ├── suffixes.py            #   nominal & verbal paradigms w/ allomorphs
│   │   └── segmenter.py           #   beam-search morphological analyzer
│   ├── text/                      # normalization + CTC vocabulary
│   ├── data/                      # dataset packaging, augmentation, sample data
│   ├── decoding/                  # n-gram KN LM + CTC prefix beam search
│   ├── metrics/                   # WER/CER + morphology-aware M-WER
│   ├── training/                  # TrainConfig, collator, wav2vec2 CTC driver
│   └── inference.py               # Transcriber (greedy / beam / beam+LM)
├── scripts/                       # thin CLI wrappers over the package
│   ├── make_sample_data.py  prepare_dataset.py  train.py
│   ├── build_lm.py          evaluate.py         transcribe.py
│   └── demo_morphology.py         # dependency-free showcase
├── configs/xlsr_kazakh.yaml       # a training configuration
├── tests/                         # pure-Python unit tests (pytest)
└── docs/                          # methodology.md + references.bib
```

## install

```bash
# pure-Python core only (morphology / metrics / LM / decoding / text):
pip install -e .

# full training + inference stack (torch, transformers, datasets, ...):
pip install -e ".[train]"

# or, classic:
pip install -r requirements.txt
```

## quickstart

### 1. see the interesting parts immediately (no GPU, no deps)

```bash
python scripts/demo_morphology.py     # morphological analysis + M-WER vs WER
pytest -q                              # run the test suite
```

### 2. end-to-end with synthetic sample data

> the bundled generator writes **synthetic demo audio** (procedural tones, *not
> speech*) so the whole pipeline runs without a real corpus. a model trained on
> it learns nothing — it exists only to exercise the plumbing.

```bash
python scripts/make_sample_data.py  --out data/sample
python scripts/prepare_dataset.py   --audio-dir data/sample --out data/prepared
python scripts/train.py             --config configs/xlsr_kazakh.yaml \
                                    --dataset data/prepared --output-dir checkpoints/demo --epochs 1
python scripts/build_lm.py          --corpus data/sample/corpus.txt --out checkpoints/kaz.lm
python scripts/evaluate.py          --model checkpoints/demo --dataset data/prepared \
                                    --strategy beam+lm --lm checkpoints/kaz.lm
```

### 3. train for real

point the same scripts at a directory of real Kazakh `.wav` + transcript pairs
(sidecar `.txt` files or a `metadata.csv` with `file_name,transcript`), keep the
`facebook/wav2vec2-large-xlsr-53` base checkpoint, and train on a GPU. build the
LM from a Kazakh text corpus (one sentence per line).

## data

no speech corpus ships with this repository. the pipeline reads either sidecar
`.txt` transcripts next to each `.wav`, or a `metadata.csv`/`.tsv` with
`file_name` and `transcript` columns; audio is resampled to 16 kHz mono.
`scripts/make_sample_data.py` generates clearly-labeled synthetic demo data.

## a note on numbers

this README quotes **no WER/accuracy figures and no exact dataset sizes**.
reported outcomes from the pilot (rural-school deployment, ~170 active users,
small government grant) are stated as-is; anything illustrative — the sample
data, the M-WER example, the demo output — is labeled illustrative. benchmark
numbers depend entirely on the corpus you train on, so publishing a specific
figure here would be meaningless at best and misleading at worst.

## references

key prior work (full BibTeX in [`docs/references.bib`](docs/references.bib)):
wav2vec 2.0 (Baevski et al. 2020), XLSR (Conneau et al. 2021), CTC
(Graves et al. 2006), prefix beam search (Hannun et al. 2014), modified
Kneser-Ney (Chen & Goodman 1998), SpecAugment (Park et al. 2019), and
finite-state Kypchak morphology (Washington et al. 2014).

## license

Apache-2.0. research code, provided as-is.
