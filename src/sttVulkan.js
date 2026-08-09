// GPU speech-to-text via whisper.cpp + Vulkan, with LIVE rolling partials.
//
// Whisper is offline/batch, so true word-by-word streaming isn't native -- we
// fake it the standard way: keep buffering speech, re-decode the in-flight
// utterance every ~1.5–2s, and emit the growing transcript as `partial`.
// On trailing silence we do one higher-quality "final" decode and commit.
//
// A long-lived Python ctypes worker keeps large-v3-turbo warm on the GPU.
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const { SAMPLE_RATE } = require('./vadEnergy');
const { spawnPython } = require('./python');
const { env } = require('./env');

const DEFAULT_BIN_DIR = path.join(__dirname, '..', 'bin', 'whisper');
const DEFAULT_MODEL = path.join(__dirname, '..', 'models', 'ggml-large-v3-turbo.bin');
const BIN_DIR = env('WHISPER_BIN', DEFAULT_BIN_DIR);
const MODEL_PATH = env('WHISPER_MODEL', DEFAULT_MODEL);
const WORKER_SCRIPT = path.join(__dirname, '..', 'whisper_worker.py');
const LANGUAGE = env('WHISPER_LANG', 'auto');
const INITIAL_PROMPT =
  env(
    'WHISPER_PROMPT',
    'Live multilingual meeting. Transcribe accurately in the spoken language. ' +
      'Short English commands: start, save, load, record, begin, stop.'
  );

// Live partial cadence — meeting defaults skip partials (they steal GPU from finals).
const PARTIAL_EVERY_MS = Number(env('PARTIAL_MS', '2000')) || 2000;
const MIN_PARTIAL_AUDIO_MS = Number(env('MIN_PARTIAL_MS', '500')) || 500;
const SILENCE_FINAL_MS = Number(env('VAD_SILENCE_MS', '480')) || 480;
// Short enough to stay live, long enough for accented phrases.
const MAX_UTTERANCE_MS = Number(env('VAD_MAX_MS', '5500')) || 5500;
const ENERGY_GATE = Number(env('VAD_THRESHOLD', '0.005')) || 0.005;
const ENABLE_PARTIALS = !/^(0|off|false|no)$/i.test(String(env('PARTIALS', '0')));
// When force-cutting a long utterance, keep this much tail for the next chunk.
const OVERLAP_MS = Number(env('VAD_OVERLAP_MS', '400')) || 400;
// Ignore Discord compressor blips shorter than this before opening an utterance.
const MIN_SPEECH_MS = Number(env('VAD_MIN_SPEECH_MS', '280')) || 280;

const MODEL_URL =
  'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin';
const BIN_URL =
  'https://github.com/eviscerations/whisper-windows-mcp/releases/download/v1.4.0/whisper-vulkan-win-x64.zip';

function assertModelPresent(modelPath = MODEL_PATH) {
  const missing = [];
  if (!fs.existsSync(path.join(BIN_DIR, 'whisper.dll'))) {
    missing.push(`whisper.dll under ${BIN_DIR}`);
  }
  if (!fs.existsSync(path.join(BIN_DIR, 'ggml-vulkan.dll'))) {
    missing.push(`ggml-vulkan.dll under ${BIN_DIR}`);
  }
  if (!fs.existsSync(modelPath)) missing.push(modelPath);
  if (!fs.existsSync(WORKER_SCRIPT)) missing.push(WORKER_SCRIPT);
  if (missing.length === 0) return;

  throw new Error(
    `Vulkan Whisper STT is not set up yet. Missing:\n` +
      missing.map((m) => `  - ${m}`).join('\n') +
      `\n\nRun once:\n` +
      `  powershell -ExecutionPolicy Bypass -File scripts/setup-stt-gpu.ps1\n` +
      `\nOr point RCLI_XL8_WHISPER_BIN / RCLI_XL8_WHISPER_MODEL at an existing install.\n` +
      `\nOr download manually:\n` +
      `  binaries: ${BIN_URL}\n` +
      `  model:    ${MODEL_URL}`
  );
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function mergeChunks(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/**
 * Decode queue: partials coalesce to the latest audio (drop stale).
 * Finals are FIFO (never overwrite an unresolved final — that lost captions).
 * Meeting finals can pass priority > 0 to jump ahead of mic finals.
 */
function createDecodeScheduler(decodeFn) {
  let busy = false;
  let latestPartial = null; // { samples, prompt, language, mode, resolve, reject }
  const finalQueue = []; // FIFO of final jobs

  function nextJob() {
    if (finalQueue.length) {
      // Higher priority first, else FIFO.
      let best = 0;
      for (let i = 1; i < finalQueue.length; i++) {
        if ((finalQueue[i].priority || 0) > (finalQueue[best].priority || 0)) best = i;
      }
      return finalQueue.splice(best, 1)[0];
    }
    if (latestPartial) {
      const job = latestPartial;
      latestPartial = null;
      return job;
    }
    return null;
  }

  async function pump() {
    if (busy) return;
    busy = true;
    try {
      let job;
      while ((job = nextJob())) {
        try {
          job.resolve(await decodeFn(job.samples, job.prompt, job.mode, job.language));
        } catch (err) {
          job.reject(err);
        }
      }
    } finally {
      busy = false;
      if (latestPartial || finalQueue.length) void pump();
    }
  }

  return {
    partial(samples, prompt, language) {
      return new Promise((resolve, reject) => {
        if (latestPartial) latestPartial.resolve(null); // superseded
        latestPartial = { samples, prompt, language, mode: 'partial', resolve, reject };
        void pump();
      });
    },
    final(samples, prompt, language, { priority = 0 } = {}) {
      return new Promise((resolve, reject) => {
        finalQueue.push({
          samples,
          prompt,
          language,
          mode: 'final',
          priority,
          resolve,
          reject,
        });
        // Cap backlog: keep newest high-priority + one other.
        while (finalQueue.length > 3) {
          const dropped = finalQueue.shift();
          dropped.resolve({ lang: 'und', text: '', rawText: '', dropped: true });
        }
        void pump();
      });
    },
  };
}

/** Strip common Whisper hallucinations on silence / music beds / loops. */
function scrubHallucination(text) {
  let t = String(text || '').trim();
  if (!t) return '';
  const lower = t.toLowerCase();
  const junk = [
    /^thanks? for watching\.?$/i,
    /^thank you\.?$/i,
    /^thanks?\.?$/i,
    /^please subscribe\.?$/i,
    /^subscribe to .+ channel\.?$/i,
    /^like and subscribe\.?$/i,
    /^clear conversation\.?$/i,
    /^indian english meeting\.?$/i,
    /^transcribe the spoken english accurately\.?$/i,
    /^speakers use indian english accents?\.?$/i,
    /^live multilingual meeting.*$/i,
    /^casual discord\/meet chat.*$/i,
    /^you$/,
    /^\.+$/,
    /^\[.*\]$/,
    /^\(.*\)$/,
    /^♪+$/,
    /^music$/i,
  ];
  if (junk.some((re) => re.test(t))) return '';
  if (t.length < 2) return '';
  if (/^(uh+|um+|ah+|hmm+|mm+)$/i.test(lower)) return '';

  // Collapse obvious phrase loops: "foo. foo. foo." → "foo."
  t = t.replace(/(.{8,80}?)(?:\s*\1){2,}/gi, '$1');
  // Drop if still mostly the same 3–6 word phrase repeated.
  const words = t.split(/\s+/).filter(Boolean);
  if (words.length >= 9) {
    const chunk = words.slice(0, 4).join(' ').toLowerCase();
    const joined = words.join(' ').toLowerCase();
    const repeats = joined.split(chunk).length - 1;
    if (chunk.length > 8 && repeats >= 3) return '';
  }
  // Tiny all-punctuation / emoji only
  if (!/[\p{L}\p{N}]/u.test(t)) return '';
  return t.trim();
}

/** Drop per-utterance Whisper spam that was overwriting live captions. */
function filterWhisperStderr(text) {
  return String(text || '')
    .split(/\r?\n/)
    .filter((line) => {
      if (!line.trim()) return false;
      if (/auto-detected language/i.test(line)) return false;
      if (/whisper_full_with_state/i.test(line)) return false;
      if (/whisper_full:/i.test(line)) return false;
      return true;
    })
    .join('\n');
}

function createSTTEngine(modelPath = MODEL_PATH) {
  assertModelPresent(modelPath);

  let stderrTail = '';
  let readyResolve;
  let readyReject;
  const ready = new Promise((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });
  let readySettled = false;
  const markReady = () => {
    if (readySettled) return;
    readySettled = true;
    readyResolve();
  };
  const markFailed = (err) => {
    if (readySettled) return;
    readySettled = true;
    readyReject(err);
  };

  const proc = spawnPython(
    [
      WORKER_SCRIPT,
      '--bin-dir',
      BIN_DIR,
      '--model',
      modelPath,
      '--language',
      LANGUAGE,
      '--prompt',
      INITIAL_PROMPT,
    ],
    {
      cwd: path.join(__dirname, '..'),
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: { ...process.env },
    }
  );

  proc.on('error', (err) => {
    const hint =
      err.code === 'ENOENT'
        ? `\n  Set RCLI_XL8_PYTHON (or RCLI_MEET_PYTHON) to your real python.exe if needed.`
        : '';
    markFailed(new Error(`could not start whisper_worker.py: ${err.message}${hint}`));
  });

  proc.stderr.on('data', (chunk) => {
    const text = String(chunk);
    stderrTail = (stderrTail + text).slice(-4000);
    const visible = filterWhisperStderr(text);
    if (visible.trim()) process.stdout.write(`[whisper-gpu] ${visible}`);
    if (/\[whisper-worker\] ready/i.test(text)) markReady();
    if (/failed|error/i.test(text) && /whisper_init|failed to get context/i.test(text)) {
      markFailed(new Error(text.trim()));
    }
  });

  proc.on('exit', (code, signal) => {
    if (!readySettled) {
      const detail = stderrTail.trim() ? `\n${stderrTail.trim()}` : '';
      markFailed(
        new Error(`whisper_worker exited before ready (code=${code}, signal=${signal})${detail}`)
      );
    }
  });

  setTimeout(() => {
    if (!readySettled) {
      markFailed(
        new Error(
          `whisper_worker did not become ready in time.\n${stderrTail.trim() || '(no stderr)'}`
        )
      );
    }
  }, 180000).unref?.();

  let stdoutBuf = Buffer.alloc(0);
  const pending = [];

  proc.stdout.on('data', (chunk) => {
    stdoutBuf = Buffer.concat([stdoutBuf, chunk]);
    while (pending.length && stdoutBuf.length >= 4) {
      const n = stdoutBuf.readUInt32LE(0);
      if (stdoutBuf.length < 4 + n) break;
      const text = stdoutBuf.subarray(4, 4 + n).toString('utf8');
      stdoutBuf = stdoutBuf.subarray(4 + n);
      pending.shift().resolve(text.trim());
    }
  });

  /**
   * Protocol v3:
   *   uint32 n_samples | uint8 mode | uint32 prompt_len | prompt | uint32 lang_len | lang | float32 PCM
   *   n_samples == 0 → shutdown
   * Response JSON: {"lang":"xx","text":"..."}
   */
  function parseWorkerPayload(raw) {
    const s = String(raw || '').trim();
    if (!s) return { lang: 'und', text: '', rawText: '' };
    if (s.startsWith('{')) {
      try {
        const obj = JSON.parse(s);
        const rawText = String(obj.text || '');
        return {
          lang: String(obj.lang || 'und').toLowerCase(),
          text: scrubHallucination(rawText),
          rawText,
        };
      } catch {
        /* fall through */
      }
    }
    return { lang: 'und', text: scrubHallucination(s), rawText: s };
  }

  function transcribe(samples, prompt, mode, language) {
    return new Promise((resolve, reject) => {
      if (!proc.stdin.writable) {
        reject(new Error('whisper_worker stdin is closed'));
        return;
      }
      pending.push({ resolve, reject });
      const promptBuf = Buffer.from(String(prompt || ''), 'utf8');
      const langBuf = Buffer.from(String(language || LANGUAGE || 'auto'), 'utf8');
      const header = Buffer.alloc(4 + 1 + 4);
      header.writeUInt32LE(samples.length, 0);
      header.writeUInt8(mode === 'final' ? 1 : 0, 4);
      header.writeUInt32LE(promptBuf.length, 5);
      const langHeader = Buffer.alloc(4);
      langHeader.writeUInt32LE(langBuf.length, 0);
      const pcm = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      try {
        proc.stdin.write(header);
        if (promptBuf.length) proc.stdin.write(promptBuf);
        proc.stdin.write(langHeader);
        if (langBuf.length) proc.stdin.write(langBuf);
        proc.stdin.write(pcm);
      } catch (err) {
        pending.pop();
        reject(err);
      }
    });
  }

  let engineClosed = false;
  const openStreams = new Set();
  // Shared across meeting + mic streams so GPU isn't double-booked.
  const scheduler = createDecodeScheduler(async (samples, prompt, mode, language) => {
    await ready;
    let energy = 0;
    for (let i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
    if (samples.length < SAMPLE_RATE * 0.25 || energy / samples.length < 1e-7) {
      return { lang: 'und', text: '' };
    }
    const raw = await transcribe(samples, prompt, mode, language);
    return parseWorkerPayload(raw);
  });

  /**
   * @param {{language?: string, prompt?: string}} [streamOpts]
   */
  function createStream(streamOpts = {}) {
    if (engineClosed) throw new Error('createStream: STT engine is already closed');
    const emitter = new EventEmitter();
    let streamClosed = false;
    const streamLanguage = streamOpts.language || LANGUAGE || 'auto';
    const streamPrompt = streamOpts.prompt || INITIAL_PROMPT;
    const wantPartials = streamOpts.partials != null ? !!streamOpts.partials : ENABLE_PARTIALS;
    const decodePriority = Number(streamOpts.priority) || 0;

    const chunks = [];
    let bufferedMs = 0;
    let silenceMs = 0;
    let loudMs = 0;
    let speaking = false;
    let lastPartialAt = 0;
    let lastPartialText = '';
    let committedPrompt = ''; // prior finals -- conditions Whisper on meeting context
    let decodeGen = 0; // ignore stale async results after reset
    // Soft hint only — never force sticky lang when user asked for auto (mid-call switches).
    let lastHeardLang = streamLanguage !== 'auto' ? streamLanguage : '';
    let noiseFloor = ENERGY_GATE * 0.4;

    function resetUtterance() {
      chunks.length = 0;
      bufferedMs = 0;
      silenceMs = 0;
      loudMs = 0;
      speaking = false;
      lastPartialAt = 0;
      lastPartialText = '';
      decodeGen++;
    }

    function contextPrompt() {
      const tail = committedPrompt.slice(-120);
      const hint =
        streamLanguage === 'auto' && lastHeardLang
          ? `(previous utterance sounded like ${lastHeardLang})`
          : '';
      return [streamPrompt, hint, tail].filter(Boolean).join(' ');
    }

    function decodeLanguage() {
      if (streamLanguage && streamLanguage !== 'auto') return streamLanguage;
      return 'auto';
    }

    function snapshot() {
      return mergeChunks(chunks);
    }

    function keepOverlapTail() {
      if (OVERLAP_MS <= 0 || !chunks.length) {
        chunks.length = 0;
        bufferedMs = 0;
        return;
      }
      const keepSamples = Math.floor((OVERLAP_MS / 1000) * SAMPLE_RATE);
      const all = mergeChunks(chunks);
      if (all.length <= keepSamples) return;
      const tail = all.subarray(all.length - keepSamples);
      chunks.length = 0;
      chunks.push(Float32Array.from(tail));
      bufferedMs = OVERLAP_MS;
    }

    function adaptiveGate() {
      return Math.max(ENERGY_GATE, Math.min(0.04, noiseFloor * 3.2));
    }

    function requestPartial() {
      if (!wantPartials) return;
      if (streamClosed || engineClosed || !speaking) return;
      if (bufferedMs < MIN_PARTIAL_AUDIO_MS) return;
      const gen = decodeGen;
      const samples = snapshot();
      const started = Date.now();
      lastPartialAt = started;
      scheduler
        .partial(samples, contextPrompt(), decodeLanguage())
        .then((result) => {
          if (streamClosed || engineClosed || gen !== decodeGen) return;
          if (result == null) return; // superseded
          const text = result.text || '';
          if (!text || text === lastPartialText) return;
          lastPartialText = text;
          if (result.lang && result.lang !== 'und') lastHeardLang = result.lang;
          emitter.emit('partial', text, result.lang || 'und');
        })
        .catch((err) => {
          if (!streamClosed && !engineClosed) emitter.emit('error', err);
        });
    }

    function requestFinal({ forceCut = false } = {}) {
      if (streamClosed || engineClosed) return;
      if (bufferedMs < 200) {
        resetUtterance();
        return;
      }
      const samples = snapshot();
      const prompt = contextPrompt();
      const lang = decodeLanguage();
      const wallStartedAt = Date.now() - Math.round(bufferedMs);
      // Bump generation so any in-flight partial result is ignored.
      decodeGen++;
      if (forceCut) keepOverlapTail();
      else {
        chunks.length = 0;
        bufferedMs = 0;
      }
      silenceMs = 0;
      speaking = forceCut; // stay in speaking state across a hard cut
      if (!forceCut) loudMs = 0;
      lastPartialAt = 0;
      lastPartialText = '';

      scheduler
        .final(samples, prompt, lang, { priority: decodePriority })
        .then((result) => {
          if (streamClosed || engineClosed) return;
          lastPartialText = '';
          const t = (result && result.text) || '';
          const detected = (result && result.lang) || 'und';
          const rawText = (result && result.rawText) || t;
          if (t) {
            committedPrompt = t.slice(-120);
            if (detected && detected !== 'und') lastHeardLang = detected;
          }
          emitter.emit('final', {
            text: t,
            lang: detected,
            rawText,
            msAudio: Math.round((samples.length / SAMPLE_RATE) * 1000),
            wallStartedAt,
          });
        })
        .catch((err) => {
          if (!streamClosed && !engineClosed) emitter.emit('error', err);
        });
    }

    emitter.sampleRate = SAMPLE_RATE;
    emitter.feed = function feed(samples) {
      if (streamClosed || engineClosed) return;
      const durMs = (samples.length / SAMPLE_RATE) * 1000;
      const level = rms(samples);
      if (!speaking) {
        // Track ambient noise so Discord compressor beds don't open utterances.
        noiseFloor = noiseFloor * 0.97 + level * 0.03;
      }
      const loud = level > adaptiveGate();

      if (loud) {
        chunks.push(samples);
        bufferedMs += durMs;
        loudMs += durMs;
        silenceMs = 0;

        if (!speaking) {
          if (loudMs < MIN_SPEECH_MS) return; // hold prebuffer; ignore compressor blips
          speaking = true;
          lastPartialAt = Date.now();
          emitter.emit('speech-start');
        }

        const now = Date.now();
        if (now - lastPartialAt >= PARTIAL_EVERY_MS) {
          requestPartial();
        }
      } else if (speaking) {
        chunks.push(samples);
        bufferedMs += durMs;
        silenceMs += durMs;
        const now = Date.now();
        if (now - lastPartialAt >= PARTIAL_EVERY_MS && silenceMs < SILENCE_FINAL_MS) {
          requestPartial();
        }
        if (silenceMs >= SILENCE_FINAL_MS) {
          requestFinal();
          return;
        }
      } else {
        // Quiet before speech-start — discard noise prebuffer.
        chunks.length = 0;
        bufferedMs = 0;
        loudMs = 0;
        silenceMs = 0;
      }

      if (speaking && bufferedMs >= MAX_UTTERANCE_MS) {
        requestFinal({ forceCut: true });
      }
    };

    emitter.close = function close() {
      if (streamClosed) return;
      streamClosed = true;
      openStreams.delete(emitter);
      if (speaking && bufferedMs > 200) requestFinal();
      else resetUtterance();
    };

    /** Drop in-flight audio without emitting a final (used when leaving record mode). */
    emitter.reset = function reset() {
      resetUtterance();
    };

    openStreams.add(emitter);
    return emitter;
  }

  return {
    createStream,
    ready: () => ready,
    close() {
      if (engineClosed) return;
      engineClosed = true;
      for (const s of Array.from(openStreams)) s.close();
      try {
        const header = Buffer.alloc(9);
        header.writeUInt32LE(0, 0);
        if (proc.stdin.writable) proc.stdin.end(header);
      } catch {
        /* ignore */
      }
      try {
        proc.kill();
      } catch {
        /* ignore */
      }
    },
  };
}

module.exports = {
  createSTTEngine,
  createDecodeScheduler,
  assertModelPresent,
  scrubHallucination,
  filterWhisperStderr,
  SAMPLE_RATE,
  MODEL_PATH,
  BIN_DIR,
  PARTIAL_EVERY_MS,
  MAX_UTTERANCE_MS,
  SILENCE_FINAL_MS,
};
