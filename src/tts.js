// Offline text-to-speech via native sherpa-onnx-node piper/VITS.
// Picks a voice matching the TARGET language (English Piper cannot speak Hindi).
// Playback uses a long-lived Python process so we don't pay startup per utterance.
const fs = require('fs');
const path = require('path');
const { spawnPython } = require('./python');
const sherpa_onnx = require('./sherpa');

const MODELS_ROOT = path.join(__dirname, '..', 'models');
const PLAY_SERVER = path.join(__dirname, '..', 'play_audio_server.py');
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
  let server = null;
  let serverReady = null;
  let fallbackProc = null;
  let playChain = Promise.resolve();

  function ensureServer(sampleRate) {
    if (server && !server.killed) return serverReady;
    const args = [PLAY_SERVER, String(sampleRate)];
    if (playbackDevice) args.push('--device', playbackDevice);
    server = spawnPython(args, { stdio: ['pipe', 'ignore', 'pipe'] });
    serverReady = new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      const timer = setTimeout(done, 1500);
      server.stderr.on('data', (chunk) => {
        if (/\[tts-play\] ready/i.test(String(chunk))) {
          clearTimeout(timer);
          done();
        }
      });
      server.on('exit', () => {
        server = null;
        clearTimeout(timer);
        done();
      });
      server.on('error', () => {
        server = null;
        clearTimeout(timer);
        done();
      });
    });
    return serverReady;
  }

  function playViaServer(samples, sampleRate) {
    return ensureServer(sampleRate).then(() => {
      if (!server || !server.stdin.writable) {
        return playViaFallback(samples, sampleRate);
      }
      return new Promise((resolve) => {
        const header = Buffer.alloc(4);
        header.writeUInt32LE(samples.length, 0);
        const pcm = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
        const onDrain = () => resolve();
        try {
          server.stdin.write(header);
          if (!server.stdin.write(pcm)) server.stdin.once('drain', onDrain);
          else {
            // Approximate play duration so we don't overlap too aggressively.
            const ms = Math.max(200, Math.round((samples.length / sampleRate) * 1000) + 80);
            setTimeout(resolve, ms);
          }
        } catch {
          resolve(playViaFallback(samples, sampleRate));
        }
      });
    });
  }

  function playViaFallback(samples, sampleRate) {
    return new Promise((resolve) => {
      const args = [PLAY_SCRIPT, String(sampleRate)];
      if (playbackDevice) args.push('--device', playbackDevice);
      fallbackProc = spawnPython(args, { stdio: ['pipe', 'ignore', 'pipe'] });
      const proc = fallbackProc;
      proc.on('error', () => resolve());
      proc.on('exit', () => {
        if (fallbackProc === proc) fallbackProc = null;
        resolve();
      });
      proc.stdin.write(Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength));
      proc.stdin.end();
    });
  }

  return {
    sampleRate: tts.sampleRate,
    playbackDevice,
    voiceLang: pack.lang,
    missingVoiceFor: pack.missing || null,

    synthesize(text) {
      return tts.generate({ text, sid: 0, speed: 1.25 });
    },

    speak(text) {
      if (closed || !text || !text.trim()) return Promise.resolve();
      const { samples, sampleRate } = this.synthesize(text);
      // Serialize playback so lines don't overlap into gibberish.
      playChain = playChain
        .catch(() => {})
        .then(() => {
          if (closed) return;
          return playViaServer(samples, sampleRate);
        });
      return playChain;
    },

    /** Interrupt current / queued playback (used on stop / newer line). */
    stop() {
      playChain = Promise.resolve();
      if (fallbackProc) {
        try {
          fallbackProc.kill();
        } catch {
          /* ignore */
        }
        fallbackProc = null;
      }
      if (server) {
        try {
          const header = Buffer.alloc(4);
          header.writeUInt32LE(0, 0);
          if (server.stdin.writable) server.stdin.write(header);
          server.kill();
        } catch {
          /* ignore */
        }
        server = null;
      }
    },

    close() {
      if (closed) return;
      closed = true;
      this.stop();
    },
  };
}

module.exports = { createTTS, resolveVoicePack, VOICE_PACKS };
