"""Assemble processor + model + HuggingFace Trainer for wav2vec2 CTC fine-tuning.

Heavy imports (torch / transformers / datasets) are deferred into the functions
so that importing :mod:`narkesken` stays cheap for the pure-Python components
(morphology, metrics, decoding, LM).
"""

from __future__ import annotations

from pathlib import Path

from ..text import load_vocab
from .collator import DataCollatorCTCWithPadding
from .config import TrainConfig


def build_processor(vocab_path: str | Path):
    """Create a Wav2Vec2Processor from a character ``vocab.json``."""
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )

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
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def build_model(cfg: TrainConfig, processor):
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(
        cfg.base_model,
        attention_dropout=cfg.attention_dropout,
        hidden_dropout=cfg.hidden_dropout,
        feat_proj_dropout=cfg.feat_proj_dropout,
        mask_time_prob=cfg.mask_time_prob,
        layerdrop=cfg.layerdrop,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=cfg.ctc_zero_infinity,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()
    return model


def run_training(cfg: TrainConfig) -> None:
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import Trainer, TrainingArguments

    from ..metrics import word_error_rate

    dataset_dir = Path(cfg.dataset)
    vocab_path = dataset_dir / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(f"vocab.json missing in {dataset_dir}; run prepare first")

    processor = build_processor(vocab_path)
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
    collator = DataCollatorCTCWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": word_error_rate(label_str, pred_str)}

    model = build_model(cfg, processor)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        logging_steps=cfg.logging_steps,
        group_by_length=cfg.group_by_length,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"Saved fine-tuned model + processor -> {cfg.output_dir}")
