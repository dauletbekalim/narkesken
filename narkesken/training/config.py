"""Typed training configuration with YAML/JSON loading and CLI overrides."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # data / model
    dataset: str = "data/prepared"
    output_dir: str = "checkpoints/narkesken"
    base_model: str = "facebook/wav2vec2-large-xlsr-53"

    # optimization
    epochs: float = 30.0
    batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.005
    lr_scheduler: str = "linear"          # or "cosine"
    max_grad_norm: float = 1.0

    # wav2vec2 specifics
    freeze_feature_encoder: bool = True
    attention_dropout: float = 0.05
    hidden_dropout: float = 0.05
    feat_proj_dropout: float = 0.05
    mask_time_prob: float = 0.06          # wav2vec2's built-in SpecAugment
    layerdrop: float = 0.05
    ctc_zero_infinity: bool = True

    # augmentation (our extra pipeline, applied on-the-fly)
    use_spec_augment: bool = True

    # runtime
    fp16: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 25
    group_by_length: bool = True
    seed: int = 42

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith((".yml", ".yaml")):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError as e:  # pragma: no cover
                raise SystemExit("PyYAML needed for YAML configs; use JSON or `pip install pyyaml`") from e
        else:
            data = json.loads(text)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        clean = {}
        for k, v in data.items():
            key = k.replace("-", "_")
            if key not in known:
                raise SystemExit(f"Unknown config key: {k}")
            clean[key] = v
        return cls(**clean)

    def apply_overrides(self, overrides: dict[str, Any]) -> "TrainConfig":
        """Return a copy with non-None overrides applied (for CLI merging)."""
        merged = asdict(self)
        for k, v in overrides.items():
            if v is not None and k in merged:
                merged[k] = v
        return TrainConfig(**merged)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
