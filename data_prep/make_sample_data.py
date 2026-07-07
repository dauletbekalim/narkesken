"""Generate a tiny SYNTHETIC sample dataset so the pipeline can run end-to-end.

⚠️  THIS IS SAMPLE / DEMO DATA — NOT THE REAL TRAINING SET.

    - The audio is programmatically generated tones + noise. It is NOT speech
      and contains no actual Kazakh (or any) spoken content.
    - The transcripts are short real Kazakh phrases used only so the vocabulary
      and text pipeline exercise Kazakh Cyrillic characters.
    - A model trained on this will learn nothing useful. The only purpose is to
      let ``prepare_dataset.py`` / ``train.py`` / ``evaluate.py`` run without a
      real corpus present.

The real Narkesken pilot used my own recorded Kazakh speech, which is not
included in this repository.

    python data_prep/make_sample_data.py --out data/sample --num 12
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import wave
from pathlib import Path

# Short Kazakh phrases (already valid Kazakh Cyrillic). Content is irrelevant to
# the fake audio — these only give the text pipeline something real to chew on.
SAMPLE_PHRASES = [
    "сәлеметсіз бе",
    "қайырлы таң",
    "мен мектепке барамын",
    "кітап оқып отырмын",
    "бүгін ауа райы жақсы",
    "балалар сабаққа келді",
    "менің атым нұрсұлтан",
    "су ішкім келеді",
    "үй жұмысын жаздым",
    "рахмет сізге",
    "қалың қалай",
    "ертең кездесеміз",
]

SAMPLE_RATE = 16_000
DURATION_SECONDS = 1.2


def _write_tone_wav(path: Path, freq: float, seed: int) -> None:
    """Write a mono 16 kHz 16-bit PCM WAV of a noisy tone (placeholder audio)."""
    rng = random.Random(seed)
    n_frames = int(SAMPLE_RATE * DURATION_SECONDS)
    amplitude = 0.25 * 32767

    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for i in range(n_frames):
            t = i / SAMPLE_RATE
            tone = math.sin(2 * math.pi * freq * t)
            noise = rng.uniform(-0.15, 0.15)
            sample = int(max(-1.0, min(1.0, tone + noise)) * amplitude)
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--num",
        type=int,
        default=len(SAMPLE_PHRASES),
        help="Number of sample utterances to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    metadata_rows = []
    for i in range(args.num):
        phrase = SAMPLE_PHRASES[i % len(SAMPLE_PHRASES)]
        stem = f"utt_{i:04d}"
        wav_path = out / f"{stem}.wav"
        freq = rng.uniform(120, 320)  # arbitrary; this is not speech
        _write_tone_wav(wav_path, freq=freq, seed=args.seed + i)

        # Sidecar transcript, the layout prepare_dataset.py reads by default.
        (out / f"{stem}.txt").write_text(phrase, encoding="utf-8")
        metadata_rows.append((f"{stem}.wav", phrase))

    # Also drop a metadata.csv so the other input convention is demonstrated.
    # (prepare_dataset.py prefers metadata.csv if present; remove it to test the
    # sidecar-.txt path instead.)
    meta_path = out / "metadata.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        f.write("file_name,transcript\n")
        for file_name, transcript in metadata_rows:
            f.write(f"{file_name},{transcript}\n")

    marker = out / "SAMPLE_DATA_README.txt"
    marker.write_text(
        "This directory contains SYNTHETIC SAMPLE/DEMO data only.\n"
        "The .wav files are generated tones+noise, NOT speech.\n"
        "Do not treat this as the real Narkesken training set.\n",
        encoding="utf-8",
    )

    print(f"Wrote {args.num} sample utterances to {out}")
    print(f"  (synthetic demo data — not real speech; see {marker.name})")


if __name__ == "__main__":
    main()
