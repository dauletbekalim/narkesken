"""CLI: transcribe one or more .wav files with a fine-tuned model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("wavs", nargs="+", type=Path)
    p.add_argument("--strategy", default="beam", choices=["greedy", "beam", "beam+lm"])
    p.add_argument("--lm", type=Path)
    p.add_argument("--word-lm", action="store_true")
    p.add_argument("--beam-width", type=int, default=32)
    args = p.parse_args()

    import soundfile as sf

    from narkesken.inference import Transcriber

    tr = Transcriber(args.model, lm_path=args.lm, morpheme_lm=not args.word_lm)
    for wav in args.wavs:
        audio, sr = sf.read(str(wav))
        text = tr.transcribe(audio, sampling_rate=sr, strategy=args.strategy,
                             beam_width=args.beam_width)
        print(f"{wav.name}\t{text}")


if __name__ == "__main__":
    main()
