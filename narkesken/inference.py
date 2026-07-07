"""Transcription: run a fine-tuned wav2vec2 model + choose a CTC decoder.

Supports three decoding strategies:

- ``greedy``       — best-path argmax (fast baseline).
- ``beam``         — CTC prefix beam search (no LM).
- ``beam+lm``      — CTC prefix beam search with n-gram LM shallow fusion,
                     optionally morpheme-factored (recommended for Kazakh).
"""

from __future__ import annotations

import math
from pathlib import Path

from .decoding import (
    ModifiedKneserNeyLM,
    ctc_prefix_beam_search,
    greedy_ctc_decode,
    morpheme_lm_scorer,
)
from .morphology import KazakhSegmenter


class Transcriber:
    def __init__(self, model_dir: str | Path, *, device: str | None = None,
                 lm_path: str | Path | None = None, morpheme_lm: bool = True):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
        self.model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # id<->char maps and blank id (CTC blank == pad token for wav2vec2).
        vocab = self.processor.tokenizer.get_vocab()
        self.id_to_char = {i: c for c, i in vocab.items()}
        self.blank_id = self.processor.tokenizer.pad_token_id
        self.delimiter = self.processor.tokenizer.word_delimiter_token or "|"

        self.lm = None
        self.lm_scorer = None
        if lm_path is not None:
            self.lm = ModifiedKneserNeyLM.load(lm_path)
            if morpheme_lm:
                self.lm_scorer = morpheme_lm_scorer(self.lm, KazakhSegmenter())
            else:
                self.lm_scorer = self._word_scorer(self.lm)

    @staticmethod
    def _word_scorer(lm):
        def score(history, word):
            return lm.logprob(word, history[-(lm.order - 1):])
        return score

    def _log_probs(self, audio_array, sampling_rate: int = 16_000):
        import torch

        inputs = self.processor(audio_array, sampling_rate=sampling_rate,
                                return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1).cpu()
        return log_probs.tolist()

    def transcribe(self, audio_array, *, sampling_rate: int = 16_000,
                   strategy: str = "greedy", beam_width: int = 32,
                   lm_weight: float = 0.4, word_bonus: float = 1.2) -> str:
        log_probs = self._log_probs(audio_array, sampling_rate)

        if strategy == "greedy":
            return greedy_ctc_decode(
                log_probs, self.id_to_char, blank_id=self.blank_id,
                word_delimiter=self.delimiter,
            )

        scorer = self.lm_scorer if strategy == "beam+lm" else None
        results = ctc_prefix_beam_search(
            log_probs, self.id_to_char, blank_id=self.blank_id,
            word_delimiter=self.delimiter, beam_width=beam_width,
            lm_scorer=scorer, lm_weight=lm_weight, word_bonus=word_bonus,
        )
        return results[0][0] if results else ""
