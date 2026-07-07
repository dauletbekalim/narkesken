# Narkesken

**Reference implementation of a Kazakh-language speech-recognition project I built and piloted in 2024–2025.**

The original codebase was lost. This repository reconstructs the technical approach from scratch, for documentation and reference purposes.

---

## Status

> **This is a reference / research implementation — not the original production pilot code.**
>
> The original Narkesken project (fine-tuned wav2vec2 on my own recorded Kazakh speech, piloted in rural Kazakh-speaking schools) was built ~2 years ago and its code no longer exists. This repo rebuilds the *same approach* — data prep, CTC fine-tuning of wav2vec2, WER evaluation — as clean, runnable reference code. It does **not** ship the original trained model or the real training dataset, and the scripts here are meant to be read and adapted rather than treated as a turnkey production pipeline.

---

## Problem statement

Kazakh is a **low-resource, agglutinative** language, and both properties make automatic speech recognition (ASR) hard:

- **Agglutination.** A single Kazakh root can take dozens of suffixed surface forms — case, number, possession, tense, and person all stack as suffixes on the root. So the effective vocabulary a model must cover is far larger than the set of dictionary roots, and rare inflected forms are everywhere in ordinary speech.
- **Data scarcity.** Compared to English or Russian, there is very little transcribed Kazakh speech, and very little tooling built specifically for it.
- **Model mismatch.** Most readily available speech models are trained on English or Russian. In a Kazakh-speaking region (including bilingual Kazakh/Russian settings), those models systematically mishear Kazakh phonology and morphology.

This is the classic low-resource-language ASR situation: high morphological variation, almost no existing tooling, and off-the-shelf models trained on the wrong languages.

## Approach

Fine-tune Meta's **wav2vec2** — a self-supervised speech representation model — on Kazakh speech data using **CTC** (Connectionist Temporal Classification) loss.

- **Base checkpoint:** a multilingual XLSR checkpoint such as [`facebook/wav2vec2-large-xlsr-53`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53). XLSR is pretrained on many languages, which gives a better phonetic starting point for a low-resource target than an English-only model.
- **Head:** a character-level CTC head over a Kazakh (Cyrillic) vocabulary built from the training transcripts.
- **Pipeline:** `data_prep/` normalizes and packages `.wav` + transcript pairs into a HuggingFace `datasets` format → `train.py` fine-tunes with `transformers` + `torch` → `evaluate.py` reports WER (and a morphology-aware metric stub, below).

Character-level CTC is a deliberate choice for an agglutinative language: it does not commit to a fixed word vocabulary, so it can in principle spell out inflected forms it never saw as whole words during training.

## Outcomes from the original project (2024–2025)

These describe the real pilot, not this reference code:

- **Piloted in rural Kazakh-speaking schools.**
- **~170 active users** during the pilot.
- **Funded by a small government grant.**

I'm deliberately *not* quoting exact WER numbers or exact dataset sizes here — the original measurements are gone, and I won't reconstruct specific figures from memory. The approach and outcomes above are accurate; anything numeric beyond them would be a guess, so it's left out.

## Known limitation & future direction: morphology-aware scoring

Standard **Word Error Rate (WER)** treats every wrong word as equally wrong. For an agglutinative language that is misleading in a specific, important way:

- If the model gets the **root** right but attaches the **wrong suffix**, WER counts a full word error — the same penalty as producing a completely unrelated word. But a wrong suffix usually means the *root meaning was recognized* and only the grammatical marking (case, tense, possession, number) is off.
- Conversely, getting the **root wrong** is a more fundamental failure than a suffix slip, yet WER can't tell the two apart.

So a single WER number **understates** how a model actually behaves on Kazakh: it hides whether errors are "understood the word, botched the grammar" versus "didn't recognize the word at all." Those have very different downstream consequences (e.g. for a reading/pronunciation tutor used by schoolchildren).

**Proposed direction:** a **morphology-aware metric** that splits each word into a root and a suffix chain and weights the two kinds of error differently — a smaller penalty for a correct root with a wrong suffix chain, a larger penalty for a wrong root. A runnable starting point (a real, deliberately simple segmentation + weighting, clearly marked as a starting point rather than a validated linguistic model) lives in [`evaluate.py`](evaluate.py) as `morphology_aware_error_rate`.

This is explicitly *future work*: proper Kazakh morphological segmentation is its own problem, and the stub uses a crude heuristic so the idea is expressed in code, not just prose.

---

## Repository layout

```
narkesken/
├── README.md
├── requirements.txt
├── .gitignore
├── data_prep/
│   ├── normalize.py        # Kazakh Cyrillic text normalization
│   ├── prepare_dataset.py  # .wav + transcript pairs -> HuggingFace datasets format
│   └── make_sample_data.py # generate small SAMPLE/DEMO data (not the real dataset)
├── train.py                # wav2vec2 CTC fine-tuning
└── evaluate.py             # WER + morphology-aware metric stub
```

## Quickstart (with sample/demo data)

> The bundled data is **synthetic sample data for demonstration only** — it is not Kazakh speech and will not train a usable model. It exists so the pipeline runs end-to-end.

```bash
pip install -r requirements.txt

# 1. Generate small synthetic sample data (clearly labeled as demo, not the real set)
python data_prep/make_sample_data.py --out data/sample

# 2. Package .wav + transcript pairs into a HuggingFace datasets directory
python data_prep/prepare_dataset.py --audio-dir data/sample --out data/prepared

# 3. Fine-tune wav2vec2 (defaults are tiny/CPU-friendly for a smoke test)
python train.py --dataset data/prepared --output-dir checkpoints/demo --epochs 1

# 4. Evaluate: standard WER + morphology-aware stub
python evaluate.py --model checkpoints/demo --dataset data/prepared
```

To train for real, point the same scripts at a directory of real Kazakh `.wav` + transcript pairs and use a full base checkpoint (e.g. `facebook/wav2vec2-large-xlsr-53`) on a GPU.

## License

Reference/research code. Provided as-is for documentation and educational purposes.
