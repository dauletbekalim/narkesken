"""Fine-tune wav2vec2 for Kazakh ASR with CTC loss.

This mirrors the approach used in the original Narkesken pilot: take a
multilingual wav2vec2 checkpoint (XLSR) and fine-tune a character-level CTC head
on Kazakh speech prepared by ``data_prep/prepare_dataset.py``.

    python train.py --dataset data/prepared --output-dir checkpoints/demo

The defaults are deliberately tiny so the script runs as a CPU smoke test on the
synthetic sample data. For a real run, pass a full base checkpoint
(``--base-model facebook/wav2vec2-large-xlsr-53``), a GPU, and realistic
hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Prepared dataset directory from prepare_dataset.py (with vocab.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write checkpoints + processor.",
    )
    parser.add_argument(
        "--base-model",
        default="facebook/wav2vec2-large-xlsr-53",
        help="Base wav2vec2 checkpoint to fine-tune (default: XLSR-53).",
    )
    parser.add_argument("--epochs", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument(
        "--freeze-feature-encoder",
        action="store_true",
        default=True,
        help="Freeze the conv feature encoder (standard for XLSR fine-tuning).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file whose keys override the above arguments.",
    )
    args = parser.parse_args()

    # Let a JSON config override CLI defaults (config wins over argparse defaults,
    # but explicit CLI flags you pass still take precedence over the file only if
    # you also set them; for simplicity here the file overrides everything it names).
    if args.config is not None:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for key, value in overrides.items():
            key = key.replace("-", "_")
            if not hasattr(args, key):
                raise SystemExit(f"Unknown config key: {key}")
            setattr(args, key, value)
    return args


@dataclass
class DataCollatorCTCWithPadding:
    """Pad input audio features and label sequences independently for CTC.

    Standard collator for wav2vec2 CTC fine-tuning: the processor pads the raw
    audio to the longest item in the batch, labels are padded separately, and
    padding positions in the labels are set to ``-100`` so CTC loss ignores them.
    """

    processor: Any
    padding: bool = True

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()

    # Heavy imports are deferred so --help works without the full stack installed.
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    dataset_dir: Path = args.dataset
    vocab_path = dataset_dir / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(
            f"vocab.json not found in {dataset_dir}. "
            "Run data_prep/prepare_dataset.py first."
        )

    # --- Processor (tokenizer + feature extractor) -------------------------
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # --- Dataset -----------------------------------------------------------
    ds = load_from_disk(str(dataset_dir))

    def prepare_example(batch: dict) -> dict:
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    ds = ds.map(prepare_example, remove_columns=ds["train"].column_names)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # --- Metric (WER during eval) -----------------------------------------
    import evaluate as hf_evaluate

    wer_metric = hf_evaluate.load("wer")

    def compute_metrics(pred) -> dict:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    # --- Model -------------------------------------------------------------
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # --- Training ----------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        group_by_length=True,
        report_to=[],  # no external loggers by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved fine-tuned model + processor to {args.output_dir}")


if __name__ == "__main__":
    main()
