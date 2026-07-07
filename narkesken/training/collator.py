"""CTC data collator: independently pad audio inputs and label sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorCTCWithPadding:
    """Standard wav2vec2 CTC collator.

    The processor pads the raw audio to the longest item in the batch; labels
    are padded separately and their padding positions are set to ``-100`` so the
    CTC loss ignores them.
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
