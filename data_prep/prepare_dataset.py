"""Package a directory of ``.wav`` + transcript pairs into a HuggingFace
``datasets`` directory that ``train.py`` / ``evaluate.py`` can load.

Expected input layout (one transcript per audio file). Two conventions are
supported:

1. Sidecar ``.txt`` files:

       data/sample/
         utt_0001.wav
         utt_0001.txt   # contains the transcript for utt_0001.wav
         utt_0002.wav
         utt_0002.txt

2. A single ``metadata.csv`` / ``metadata.tsv`` with columns ``file_name`` and
   ``transcript`` (relative paths under ``--audio-dir``).

Transcripts are normalized with ``data_prep/normalize.py`` and a character
vocabulary (``vocab.json``) is written next to the dataset for the CTC head.

    python data_prep/prepare_dataset.py --audio-dir data/sample --out data/prepared
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Support running both as a module and as a script.
try:
    from .normalize import build_vocab, normalize_text
except ImportError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from normalize import build_vocab, normalize_text  # type: ignore


TARGET_SAMPLING_RATE = 16_000  # wav2vec2 expects 16 kHz mono audio.


def _load_pairs_from_metadata(meta_path: Path, audio_dir: Path) -> list[dict]:
    delimiter = "\t" if meta_path.suffix.lower() == ".tsv" else ","
    pairs: list[dict] = []
    with meta_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None or "file_name" not in reader.fieldnames:
            raise ValueError(
                f"{meta_path} must have a header with at least "
                "'file_name' and 'transcript' columns."
            )
        for row in reader:
            wav_path = (audio_dir / row["file_name"]).resolve()
            pairs.append({"audio_path": str(wav_path), "text": row.get("transcript", "")})
    return pairs


def _load_pairs_from_sidecars(audio_dir: Path) -> list[dict]:
    pairs: list[dict] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"  ! skipping {wav_path.name}: no matching {txt_path.name}")
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        pairs.append({"audio_path": str(wav_path.resolve()), "text": text})
    return pairs


def collect_pairs(audio_dir: Path) -> list[dict]:
    """Find (audio_path, transcript) pairs via metadata file or sidecar .txt."""
    for meta_name in ("metadata.csv", "metadata.tsv"):
        meta_path = audio_dir / meta_name
        if meta_path.exists():
            print(f"Reading transcripts from {meta_path}")
            return _load_pairs_from_metadata(meta_path, audio_dir)

    print(f"Reading transcripts from sidecar .txt files in {audio_dir}")
    return _load_pairs_from_sidecars(audio_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        required=True,
        type=Path,
        help="Directory containing .wav files + transcripts (sidecar .txt or metadata.csv).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for the HuggingFace datasets dump (+ vocab.json).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction held out for the test split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for the train/test split.",
    )
    args = parser.parse_args()

    # Imported lazily so that --help works without the heavy deps installed.
    from datasets import Audio, Dataset

    audio_dir: Path = args.audio_dir
    if not audio_dir.is_dir():
        raise SystemExit(f"--audio-dir not found: {audio_dir}")

    pairs = collect_pairs(audio_dir)
    if not pairs:
        raise SystemExit(f"No .wav + transcript pairs found under {audio_dir}")

    # Normalize transcripts up front.
    for p in pairs:
        p["text"] = normalize_text(p["text"])
    pairs = [p for p in pairs if p["text"]]  # drop rows that normalized to empty
    print(f"Collected {len(pairs)} usable utterances.")

    # Build the character vocabulary from the normalized transcripts.
    vocab = build_vocab([p["text"] for p in pairs])

    ds = Dataset.from_list(pairs)
    # Decode audio lazily at the sampling rate wav2vec2 expects.
    ds = ds.cast_column("audio_path", Audio(sampling_rate=TARGET_SAMPLING_RATE))
    ds = ds.rename_column("audio_path", "audio")

    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(out))

    vocab_path = out / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset to {out}")
    print(f"  train: {len(split['train'])}  test: {len(split['test'])}")
    print(f"  vocab: {vocab_path} ({len(vocab)} tokens)")


if __name__ == "__main__":
    main()
