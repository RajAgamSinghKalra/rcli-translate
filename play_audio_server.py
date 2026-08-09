"""Persistent PCM playback server for rcli-translate TTS.

Protocol (little-endian):
  request:  uint32 n_samples | float32[n_samples]   (n_samples==0 → quit)
  no response

Keeps the soundcard speaker handle open so each utterance skips Python/startup cost.
"""
from __future__ import annotations

import argparse
import struct
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


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed")
        buf.extend(chunk)
    return bytes(buf)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_rate", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    speaker = resolve_speaker(args.device)
    sys.stderr.write(f"[tts-play] ready: {speaker.name}\n")
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    try:
        while True:
            header = read_exact(stdin, 4)
            (n_samples,) = struct.unpack("<I", header)
            if n_samples == 0:
                break
            raw = read_exact(stdin, n_samples * 4)
            samples = np.frombuffer(raw, dtype=np.float32)
            if samples.size:
                speaker.play(samples, samplerate=args.sample_rate)
    except EOFError:
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
