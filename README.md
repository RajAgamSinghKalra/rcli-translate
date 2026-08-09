# rcli-translate

Offline **live meeting translator** for Windows. Captures the other person in Google Meet (or any app) via WASAPI loopback, **auto-detects their language**, repairs messy ASR with a local LLM, **translates live** into your language, optionally **speaks** the translation, then lets you **ask questions** over the bilingual transcript afterward.

Forked from [rcli-meet](https://github.com/RajAgamSinghKalra/rcli-meet). Everything runs locally — nothing leaves your machine.

## Mute their voice, hear only the translation

Windows cannot mute Meet on your headphones **and** loopback that same device.
Use a virtual cable:

1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free).
2. In Chrome / Meet, set **speaker output** to **CABLE Input**.
3. Keep your real headset as the Windows **default** playback device.
4. Run:

```bat
node src\quiet.js --mute-original --loopback CABLE --speakers Kraken --to en
```

- We capture Meet from the cable (you don't hear them).
- We speak the translation on your headset (`--speakers`).
- List names anytime: `node src\quiet.js --list-audio`

Env equivalents: `RCLI_XL8_MUTE_ORIGINAL=1`, `RCLI_XL8_LOOPBACK=CABLE`, `RCLI_XL8_SPEAKERS=Kraken`.

## What you get

1. **Live translate (default on launch)** — meeting/loopback audio → Whisper (language=`auto`) → LLM repair+translate → caption like `[00:01:12] [other/hi→en] …` + spoken English (unless `--no-tts`).
2. **Your mic** — captioned into the session (and spoken commands); not live-translated in v1.
3. **After / during pause (`stop`)** — type or speak questions about what was said; answers use the translated transcript + rolling summary.

Typical delay after they finish a sentence: **~0.7–2 seconds** when already in your language (fast path), or **~1–3 seconds** when translating (VAD + Whisper + LLM).

Live UI shows a spinner for listening / translating / speaking, colored partials, and timing on each final line.

## Requirements

- Windows 10/11, Node.js ≥ 18
- Python 3 + `pip install soundcard numpy`
- Vulkan GPU recommended (tested on AMD RX 6800 XT)
- [RunAnywhere Electron SDK](https://github.com/) `dist/` build for the local LLM (same as rcli-meet)
- Headphones strongly recommended (avoid mic bleed / TTS echo)

## Setup

```bat
npm install
npm run setup:stt-gpu
pip install soundcard numpy
copy run.example.bat run.bat
```

Edit `run.bat`:

- `RCLI_XL8_SDK_DIST` — path to `@runanywhere/electron` `dist/`
- `RCLI_XL8_LLM_PATH` — catalog id (e.g. `qwen2.5-3b`) or local GGUF
- Optional: `RCLI_XL8_PYTHON`, `RCLI_XL8_TO=en`, `RCLI_XL8_OTHER=aditya`

`RCLI_MEET_*` env vars still work as fallbacks if you already configured rcli-meet.

If you already downloaded Whisper Vulkan binaries/models for rcli-meet, point:

```bat
set RCLI_XL8_WHISPER_BIN=D:\path\to\rcli-meet\bin\whisper
set RCLI_XL8_WHISPER_MODEL=D:\path\to\rcli-meet\models\ggml-large-v3-turbo.bin
```

## Run

```bat
run.bat
:: or
node src\quiet.js --to en --other aditya
```

### Commands

| Command | Meaning |
|---------|---------|
| *(autostart)* | Live translate starts immediately |
| `start` / `record` | Resume live translate + new session folder |
| `stop` | Pause translate — ask questions |
| `save` / `load` / `add <path>` | Session persistence |
| anything else | Question over the bilingual transcript |
| `/quit` | Exit |

### Flags

- `--to <lang>` — target language (default `en`)
- `--other <name>` — label for the other person (default `other`)
- `--no-tts` — captions only
- `--no-autostart` — wait for `start`
- `--mic` / `--mic-gain` / `--no-mic` — same idea as rcli-meet

## Google Meet tips

1. Join Meet with **headphones**.
2. Launch `rcli-translate` — it auto-starts translating loopback from your default speakers.
3. When they speak another language, wait for the silence gap; you’ll see `[name/xx→en] …` and hear the English line.
4. Type `stop`, then ask e.g. `what did we decide about the pilot?`

## Architecture (v1)

```
WASAPI loopback → Whisper Vulkan (auto lang) → LLM repair+translate → captions + TTS
Mic → Whisper (en) → [you] lines / spoken commands / post-pause Q&A
```

Partials show raw ASR for snappiness; **LLM translate runs on finals only**.

## Out of scope (v1)

- Bidirectional translate of *your* mic into their language
- LLM translate on every partial token
- Non-English Piper TTS voices (captions still work for other `--to` values; spoken TTS is English-oriented)

## License

MIT
