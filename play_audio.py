"""Play raw float32 PCM (mono) from stdin through a speaker.

Usage: play_audio.py <sample_rate> [--device <substring>]
"""
import argparse
import sys

import numpy as np
import soundcard as sc


def resolve_speaker(name_substr: str | None):
    speakers = list(sc.all_speakers())
    if not speakers:
        raise RuntimeError("no speakers found")
    if not name_substr:
        return sc.default_speaker()
    needle = name_substr.strip().lower()
    matches = [s for s in speakers if needle in s.name.lower()]
    if not matches:
        available = "\n".join(f"  - {s.name}" for s in speakers)
        raise RuntimeError(f'no speaker matching "{name_substr}". Available:\n{available}')
    matches.sort(key=lambda s: len(s.name))
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_rate", type=int)
    parser.add_argument(
        "--device",
        default=None,
        help='Substring to pick playback speakers (e.g. "Kraken", "Headphones").',
    )
    args = parser.parse_args()

    raw = sys.stdin.buffer.read()
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return
    speaker = resolve_speaker(args.device)
    speaker.play(samples, samplerate=args.sample_rate)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(f"fatal: {err}", file=sys.stderr, flush=True)
        sys.exit(1)
