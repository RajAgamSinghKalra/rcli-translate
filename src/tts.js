// Offline text-to-speech via native sherpa-onnx-node piper/VITS.
// Picks a voice matching the TARGET language (English Piper cannot speak Hindi).
const fs = require('fs');
const path = require('path');
const { spawnPython } = require('./python');
const sherpa_onnx = require('./sherpa');

const MODELS_ROOT = path.join(__dirname, '..', 'models');
const PLAY_SCRIPT = path.join(__dirname, '..', 'play_audio.py');
const PLAYBACK_DEVICE =
  process.env.RCLI_XL8_SPEAKERS || process.env.RCLI_MEET_SPEAKERS || '';

/** Known packed Piper/VITS voices under models/. */
const VOICE_PACKS = {
  en: {
    dir: 'vits-piper-en_US-lessac-medium',
    onnx: 'en_US-lessac-medium.onnx',
  },
  hi: {
    dir: 'vits-piper-hi_IN-pratham-medium',
    onnx: 'hi_IN-pratham-medium.onnx',
  },
};

function resolveVoicePack(lang) {
  const code = String(lang || 'en').toLowerCase().slice(0, 2);
  const override = process.env.RCLI_XL8_TTS_MODEL_DIR || process.env.RCLI_MEET_TTS_MODEL_DIR;
  if (override) {
    const onnx = fs.readdirSync(override).find((f) => f.endsWith('.onnx') && !f.includes('json'));
    if (onnx) {
      return { dirPath: override, onnx, lang: code };
    }
  }
  const pack = VOICE_PACKS[code] || VOICE_PACKS.en;
  const dirPath = path.join(MODELS_ROOT, pack.dir);
  const onnxPath = path.join(dirPath, pack.onnx);
  if (code !== 'en' && !fs.existsSync(onnxPath)) {
    // Fall back to English voice only if Hindi pack missing — caller should warn.
    const en = VOICE_PACKS.en;
    return {
      dirPath: path.join(MODELS_ROOT, en.dir),
      onnx: en.onnx,
      lang: 'en',
      missing: code,
    };
  }
  return { dirPath, onnx: pack.onnx, lang: code };
}

function loadOfflineTts(pack) {
  return new sherpa_onnx.OfflineTts({
    model: {
      vits: {
        model: path.join(pack.dirPath, pack.onnx),
        tokens: path.join(pack.dirPath, 'tokens.txt'),
        dataDir: path.join(pack.dirPath, 'espeak-ng-data'),
      },
    },
    numThreads: 2,
    provider: 'cpu',
  });
}

/**
 * @param {{device?: string, lang?: string}} opts
 */
function createTTS(opts = {}) {
  const playbackDevice = opts.device || PLAYBACK_DEVICE;
  const pack = resolveVoicePack(opts.lang || 'en');
  const tts = loadOfflineTts(pack);
  let closed = false;

  return {
    sampleRate: tts.sampleRate,
    playbackDevice,
    voiceLang: pack.lang,
    missingVoiceFor: pack.missing || null,

    synthesize(text) {
      return tts.generate({ text, sid: 0, speed: 1.2 });
    },

    speak(text) {
      if (closed || !text || !text.trim()) return Promise.resolve();
      const { samples, sampleRate } = this.synthesize(text);
      return new Promise((resolve) => {
        const args = [PLAY_SCRIPT, String(sampleRate)];
        if (playbackDevice) args.push('--device', playbackDevice);
        const proc = spawnPython(args, {
          stdio: ['pipe', 'ignore', 'pipe'],
        });
        proc.on('error', () => resolve());
        proc.on('exit', () => resolve());
        proc.stdin.write(Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength));
        proc.stdin.end();
      });
    },

    close() {
      if (closed) return;
      closed = true;
    },
  };
}

module.exports = { createTTS, resolveVoicePack, VOICE_PACKS };
