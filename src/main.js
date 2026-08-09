#!/usr/bin/env node
// rcli-translate: offline live meeting translator + local-LLM Q&A over the
// bilingual transcript. Captures Google Meet (etc.) via WASAPI loopback,
// auto-detects language, repairs ASR + translates with a local LLM, speaks
// the translation, then lets you ask questions after. Everything stays local.
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const { env } = require('./env');
const { startCapture } = require('./capture');

const STT_ENGINE = (env('STT_ENGINE', 'vulkan') || 'vulkan').toLowerCase();
const STT_MODULES = {
  vulkan: './sttVulkan',
  sensevoice: './sttSenseVoice',
  whisper: './sttWhisper',
  zipformer: './stt',
};
if (!STT_MODULES[STT_ENGINE]) {
  console.log(
    `[rcli-translate] unknown STT_ENGINE="${STT_ENGINE}" (want: vulkan|sensevoice|whisper|zipformer)`
  );
  process.exit(2);
}
const { createSTTEngine, assertModelPresent } = require(STT_MODULES[STT_ENGINE]);
const { createTranscript, resolveSourceLabels } = require('./transcript');
const { createRetrieval } = require('./retrieval');
const { createSummarizer } = require('./summary');
const { createTTS } = require('./tts');
const { parseCommand } = require('./commands');
const { getActiveWindowTitle } = require('./activeWindow');
const { newSessionDir, saveSession, addFileToSession, listSessions, loadSession, DEFAULT_SESSIONS_DIR } =
  require('./session');
const {
  loadEngine,
  askQuestion,
  DEFAULT_LLM_PATH,
  DEFAULT_EMBEDDER_ID,
  CONTEXT_TOKEN_BUDGET,
} = require('./llm');
const { translateUtterance } = require('./translate');
const { createLogger } = require('./log');
const {
  banner,
  formatPartialLine,
  formatFinalLine,
  createStatusLine,
  fitOneRow,
  paint,
  c,
} = require('./ui');

const DEFAULT_MODEL_DIRS = {
  vulkan: path.join(__dirname, '..', 'models', 'ggml-large-v3-turbo.bin'),
  sensevoice: path.join(
    __dirname,
    '..',
    'models',
    'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17'
  ),
  whisper: path.join(__dirname, '..', 'models', 'sherpa-onnx-whisper-small.en'),
  zipformer: path.join(__dirname, '..', 'models', 'sherpa-onnx-streaming-zipformer-en-2023-06-26'),
};
const MODEL_DIR = env('STT_MODEL_DIR', DEFAULT_MODEL_DIRS[STT_ENGINE]);
const RETRIEVAL_TOP_K = 5;
const SOURCES = ['meeting', 'you'];
const FILE_CHUNK_CHARS = 1500;
const MIC_PROMPT =
  'Short English commands: start, save, load, record, begin. When the user says only "start", transcribe it as start.';

const USAGE = `rcli-translate -- live translate the other person in a meeting, then ask questions

Usage: node src/main.js [options]

Options:
  --to <lang>          YOUR language — translations are spoken/shown in this
                       (default: en). Common: en hi es fr de pt ja ko zh ar ta te
  --from <lang>        Their spoken language (default: auto = detect per utterance,
                       including mid-call switches). Override with en/hi/… if needed.
  --quality            Slower Whisper finals (best_of=3) for harder accents
  --pick-to            Interactive menu to choose --to at startup
  --minutes <n>        Recency window for Q&A (default: 20)
  --llm <id|path>      RunAnywhere LLM catalog id or local GGUF path
  --embedder <id>      RunAnywhere embedder catalog id
  --no-mic             Don't capture your microphone
  --mic <name>         Substring to pick a mic (e.g. "Razer")
  --mic-gain <n>       Software mic gain (default 1)
  --no-tts             Captions only — don't speak translations or answers
  --other <name>       Caption label for the other person (default: transcribing)
  --mute-original      Hide their voice in your ears; hear only the translation
                       (requires a virtual cable — see README)
  --loopback <name>    Speaker/virtual-cable to capture Meet from (e.g. CABLE)
  --speakers <name>    Where to play translations (e.g. Kraken / Headphones)
  --list-audio         List mics + speakers and exit
  --no-autostart       Wait for "start" instead of translating immediately
  -h, --help           Show this help

In-session:
  start / record       Begin (or resume) live translation + session capture
  stop                 (typed only) Pause translation — ask questions / voice chat
  save / load / add   Session persistence (same as rcli-meet)
  anything else        Ask a question over the bilingual transcript
  /quit                Exit

Mute-original tip: install VB-Audio Virtual Cable, set Chrome/Meet output to
"CABLE Input", run with --mute-original --loopback CABLE --speakers Kraken.
We hear them on the cable; you only hear the spoken translation on your headset.

Use headphones so your mic does not pick up translated TTS.
Latency after they finish speaking is typically ~1–3s (VAD + Whisper + LLM).
Continuous speech is cut every ~5.5s with overlap so voice stays near live (stale TTS is dropped).

Environment: RCLI_XL8_* (falls back to RCLI_MEET_*) — see README.md`;

class UsageError extends Error {}

function parseArgs(argv) {
  const opts = {
    minutes: 20,
    llmPath: DEFAULT_LLM_PATH,
    embedderId: DEFAULT_EMBEDDER_ID,
    mic: true,
    tts: true,
    micName: env('MIC', ''),
    micGain: Number(env('MIC_GAIN', '1')) || 1,
    other: env('OTHER', 'transcribing'),
    to: (env('TO', 'en') || 'en').toLowerCase(),
    // auto = Whisper detects their language per utterance (handles mid-call switches)
    from: (env('FROM', 'auto') || 'auto').toLowerCase(),
    autostart: true,
    muteOriginal: /^(1|on|true|yes)$/i.test(env('MUTE_ORIGINAL', '')),
    loopbackDevice: env('LOOPBACK', ''),
    speakersDevice: env('SPEAKERS', ''),
    listAudio: false,
    pickTo: false,
    quality: false,
  };
  const takeValue = (flag, i) => {
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) throw new UsageError(`${flag} needs a value`);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') opts.help = true;
    else if (arg === '--minutes') {
      const raw = takeValue(arg, i++);
      const minutes = Number(raw);
      if (!Number.isFinite(minutes) || minutes <= 0) {
        throw new UsageError(`--minutes must be a positive number (got "${raw}")`);
      }
      opts.minutes = minutes;
    } else if (arg === '--llm') opts.llmPath = takeValue(arg, i++);
    else if (arg === '--embedder') opts.embedderId = takeValue(arg, i++);
    else if (arg === '--no-mic') opts.mic = false;
    else if (arg === '--mic') opts.micName = takeValue(arg, i++);
    else if (arg === '--mic-gain') {
      const raw = takeValue(arg, i++);
      const gain = Number(raw);
      if (!Number.isFinite(gain) || gain <= 0) {
        throw new UsageError(`--mic-gain must be a positive number (got "${raw}")`);
      }
      opts.micGain = gain;
    } else if (arg === '--other') opts.other = takeValue(arg, i++);
    else if (arg === '--to') opts.to = takeValue(arg, i++).toLowerCase();
    else if (arg === '--from') opts.from = takeValue(arg, i++).toLowerCase();
    else if (arg === '--no-tts') opts.tts = false;
    else if (arg === '--no-autostart') opts.autostart = false;
    else if (arg === '--mute-original') opts.muteOriginal = true;
    else if (arg === '--loopback') opts.loopbackDevice = takeValue(arg, i++);
    else if (arg === '--speakers') opts.speakersDevice = takeValue(arg, i++);
    else if (arg === '--list-audio') opts.listAudio = true;
    else if (arg === '--pick-to') opts.pickTo = true;
    else if (arg === '--quality') opts.quality = true;
    else throw new UsageError(`unknown option "${arg}"`);
  }
  if (opts.quality) {
    process.env.RCLI_XL8_WHISPER_BEST_OF = process.env.RCLI_XL8_WHISPER_BEST_OF || '3';
  }
  // Mute-original needs a virtual cable capture path. Default to common names.
  if (opts.muteOriginal && !opts.loopbackDevice) {
    opts.loopbackDevice = 'CABLE';
  }
  opts.sourceLabels = resolveSourceLabels({ meeting: opts.other, you: 'you' });
  return opts;
}

function muteOriginalHelp(loopback, speakers) {
  return [
    '',
    '  mute-original mode',
    '  ------------------',
    '  Windows cannot silence Meet on your headphones AND loopback the same',
    '  device. Route Meet into a virtual cable; we listen there; you hear only TTS.',
    '',
    '  1) Install VB-Audio Virtual Cable (free): https://vb-audio.com/Cable/',
    '     (Voicemeeter also works -- use its VAIO / B1 device name.)',
    '  2) In Chrome/Meet (or Windows app volume mixer): set output to "CABLE Input".',
    '  3) Keep your headphones as the Windows DEFAULT playback device.',
    `  4) Run with:  --mute-original --loopback ${loopback || 'CABLE'} --speakers ${speakers || 'Kraken'}`,
    '  5) List devices:  node src/quiet.js --list-audio',
    '',
    `  Capturing Meet from:  ${loopback || 'CABLE'}`,
    `  Playing translation on: ${speakers || '(Windows default speakers)'}`,
    '',
  ].join('\n');
}

const TO_MENU = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'Hindi' },
  { code: 'es', label: 'Spanish' },
  { code: 'fr', label: 'French' },
  { code: 'de', label: 'German' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'zh', label: 'Chinese' },
  { code: 'ar', label: 'Arabic' },
  { code: 'ta', label: 'Tamil' },
  { code: 'te', label: 'Telugu' },
];

/** Ask the user which language to translate into (TTY only). */
async function pickTargetLanguage(current = 'en') {
  if (!process.stdin.isTTY) return current;
  const rlPick = readline.createInterface({ input: process.stdin, output: process.stdout });
  process.stdout.write('\nTranslate their speech into which language?\n');
  TO_MENU.forEach((row, i) => {
    const mark = row.code === current ? ' (current)' : '';
    process.stdout.write(`  ${String(i + 1).padStart(2)}. ${row.code}  ${row.label}${mark}\n`);
  });
  const answer = await new Promise((resolve) => {
    rlPick.question('  or type a language code (e.g. en, hi)\n> ', resolve);
  });
  rlPick.close();
  const raw = String(answer || '').trim().toLowerCase();
  if (!raw) return current;
  const idx = Number(raw) - 1;
  if (Number.isInteger(idx) && TO_MENU[idx]) return TO_MENU[idx].code;
  if (/^[a-z]{2,8}$/.test(raw)) return raw;
  return current;
}

function chunkText(text, maxChars = FILE_CHUNK_CHARS) {
  const paragraphs = text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  const chunks = [];
  let current = '';
  for (const p of paragraphs) {
    if (current && current.length + p.length + 2 > maxChars) {
      chunks.push(current);
      current = '';
    }
    current = current ? `${current}\n\n${p}` : p;
    while (current.length > maxChars) {
      chunks.push(current.slice(0, maxChars));
      current = current.slice(maxChars);
    }
  }
  if (current) chunks.push(current);
  return chunks.length ? chunks : [text.slice(0, maxChars)];
}

function normalizeFinalPayload(payload) {
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    return {
      text: String(payload.text || '').trim(),
      lang: String(payload.lang || 'und').toLowerCase(),
      rawText: String(payload.rawText || payload.text || '').trim(),
      msAudio: payload.msAudio,
      wallStartedAt: payload.wallStartedAt,
    };
  }
  return { text: String(payload || '').trim(), lang: 'und', rawText: String(payload || '').trim() };
}

async function main() {
  const log = createLogger();
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    if (err instanceof UsageError) {
      console.log(`[rcli-translate] ${err.message}\n\n${USAGE}`);
      process.exit(2);
    }
    throw err;
  }
  log.info('argv', process.argv.slice(2));
  log.info('opts', {
    to: opts.to,
    from: opts.from,
    other: opts.other,
    speakers: opts.speakersDevice,
    loopback: opts.loopbackDevice,
    muteOriginal: opts.muteOriginal,
    tts: opts.tts,
  });
  if (opts.help) {
    console.log(USAGE);
    return;
  }

  if (opts.listAudio) {
    const { spawnSync } = require('child_process');
    const { resolvePython, pythonArgs } = require('./python');
    const script = path.join(__dirname, '..', 'capture_audio.py');
    const exe = resolvePython();
    const r = spawnSync(exe, pythonArgs([script, '--list-devices']), {
      encoding: 'utf8',
      windowsHide: true,
    });
    process.stdout.write(r.stderr || r.stdout || '');
    if (r.error) {
      console.log(`[rcli-translate] could not list audio: ${r.error.message}`);
      process.exit(1);
    }
    process.exit(r.status || 0);
    return;
  }

  if (opts.pickTo) {
    opts.to = await pickTargetLanguage(opts.to);
    console.log(`[rcli-translate] translating into: ${opts.to}`);
  }

  assertModelPresent(MODEL_DIR);

  console.log('[rcli-translate] loading local engine (LLM + embedder, Vulkan)...');
  const engine = await loadEngine({ llmPath: opts.llmPath, embedderId: opts.embedderId });
  console.log(
    `[rcli-translate] LLM ready: ${opts.llmPath}` +
      (engine.disableThinking ? ' (chain-of-thought suppressed)' : '')
  );
  console.log(`[rcli-translate] embedder ready: ${opts.embedderId}`);

  console.log(
    STT_ENGINE === 'vulkan'
      ? '[rcli-translate] loading Vulkan Whisper STT (multilingual, language=auto on meeting)...'
      : `[rcli-translate] loading STT engine (${STT_ENGINE})...`
  );
  const stt = createSTTEngine(MODEL_DIR);
  if (typeof stt.ready === 'function') {
    console.log('[rcli-translate] waiting for GPU whisper worker (model load)...');
    await stt.ready();
  }
  console.log(`[rcli-translate] STT ready (engine: ${STT_ENGINE}).`);

  console.log(`[rcli-translate] debug log: ${log.latestPath}`);

  const tts = opts.tts ? createTTS({ device: opts.speakersDevice, lang: opts.to }) : null;
  console.log(
    opts.tts
      ? '[rcli-translate] TTS ready — translations spoken' +
          (tts && tts.voiceLang ? ` in ${tts.voiceLang}` : '') +
          (opts.speakersDevice ? ` on "${opts.speakersDevice}"` : '') +
          ' (generic voice, not a clone of the speaker).'
      : '[rcli-translate] TTS disabled (--no-tts).'
  );
  if (tts && tts.missingVoiceFor) {
    console.log(
      `[rcli-translate] WARNING: no TTS pack for "${tts.missingVoiceFor}" — using English voice (will sound wrong).`
    );
  }
  // Warm the playback server so the first spoken line isn't cold.
  if (tts && typeof tts.warm === 'function') {
    void tts.warm().catch(() => {});
  }
  if (opts.muteOriginal) {
    process.stdout.write(muteOriginalHelp(opts.loopbackDevice, opts.speakersDevice));
  }
  if (opts.loopbackDevice) {
    console.log(`[rcli-translate] meeting capture loopback device filter: "${opts.loopbackDevice}"`);
  }
  if (opts.mic) {
    console.log(
      '[rcli-translate] mic capture enabled' +
        (opts.micName ? ` (device filter: "${opts.micName}")` : ' (Windows default input)') +
        (opts.micGain !== 1 ? `, gain×${opts.micGain}` : '') +
        '.'
    );
  }
  console.log(
    `[rcli-translate] translating meeting (auto-detect their language) → ${opts.to}; other labeled [${opts.sourceLabels.meeting}].` +
      (opts.quality ? ' [quality Whisper]' : '')
  );

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> ',
    terminal: !!process.stdin.isTTY,
  });

  let answering = false;
  let translating = false; // LLM translate in flight
  let speaking = false;
  let muteUntil = 0;
  let ttsStartedAt = 0;
  let lastSpokenText = '';
  let recording = false; // live-translate / capture active
  let translateEpoch = 0; // bump on stop to abandon queued jobs
  let translateJobToken = 0; // bump when a newer utterance supersedes in-flight LLM
  let speakEpoch = 0; // bump to cancel/supersede TTS
  let lastMeetingAsrAt = 0;
  let sessionDir = null;
  let transcript = null;
  let retrieval = null;
  let summarizer = null;
  let awaitingLoadChoice = null;
  const deferredLines = [];
  const partials = { meeting: '', you: '' };
  const utteranceStart = { meeting: null, you: null };
  const sttStreams = {};
  const captures = {};
  const translateQueue = [];
  let translatePumpRunning = false;
  const status = createStatusLine();
  const heartbeat = setInterval(() => {
    if (!recording || answering || translating || speaking) return;
    if (!process.stdout.isTTY) return;
    const ago = lastMeetingAsrAt
      ? `${Math.max(0, Math.round((Date.now() - lastMeetingAsrAt) / 1000))}s ago`
      : 'waiting for speech';
    status.set('listen', `→ ${opts.to} · last ${ago}`);
  }, 2000);
  if (heartbeat.unref) heartbeat.unref();

  function printLine(line) {
    status.stop();
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
    process.stdout.write(line + '\n');
    rl.prompt(true);
  }

  function notify(message) {
    const line = paint(c.dim, '· ') + paint(c.teal, 'rcli') + paint(c.dim, `: ${message}`);
    if (answering || translating) deferredLines.push(line);
    else printLine(line);
  }

  function ensureSessionState() {
    if (!transcript) {
      transcript = createTranscript(sessionDir || DEFAULT_SESSIONS_DIR, {
        onError: notify,
        labels: opts.sourceLabels,
      });
    }
    if (!retrieval) retrieval = createRetrieval(engine.embedder, { onError: notify });
    if (!summarizer) {
      summarizer = createSummarizer({
        llm: engine.llm,
        disableThinking: engine.disableThinking,
        onError: notify,
        labels: opts.sourceLabels,
      });
    }
  }

  function beginSession(reason) {
    const appName = getActiveWindowTitle();
    sessionDir = newSessionDir(DEFAULT_SESSIONS_DIR, appName);
    fs.mkdirSync(sessionDir, { recursive: true });
    transcript = createTranscript(sessionDir, { onError: notify, labels: opts.sourceLabels });
    retrieval = createRetrieval(engine.embedder, { onError: notify });
    summarizer = createSummarizer({
      llm: engine.llm,
      disableThinking: engine.disableThinking,
      onError: notify,
      labels: opts.sourceLabels,
    });
    if (sttStreams.meeting && typeof sttStreams.meeting.reset === 'function') {
      sttStreams.meeting.reset();
    }
    recording = true;
    if (summarizer && typeof summarizer.setPaused === 'function') summarizer.setPaused(true);
    log.attachSession(sessionDir);
    log.info('live translate ON', { reason, appName, to: opts.to, sessionDir });
    status.set('listen', `→ ${opts.to}`);
    notify(
      `live translate ON (${reason}; window: "${appName}"; → ${opts.to}). Session: ${sessionDir}`
    );
    notify(`debug log: ${log.sessionPath || log.latestPath}`);
    if (opts.mic) {
      notify('mic STT paused while live (Meet keeps the GPU) — type "stop" to ask questions by voice.');
    }
  }

  async function handleCommand(cmd) {
    if (cmd === 'start') {
      if (recording) return notify('already translating.');
      beginSession('start');
    } else if (cmd === 'stop') {
      if (!recording) return notify('not currently translating.');
      recording = false;
      translateEpoch += 1;
      speakEpoch += 1;
      translateQueue.length = 0;
      if (tts && typeof tts.stop === 'function') tts.stop();
      if (summarizer && typeof summarizer.setPaused === 'function') {
        summarizer.setPaused(false);
        summarizer.maybeUpdate({ force: true });
      }
      if (retrieval && typeof retrieval.flush === 'function') retrieval.flush();
      if (sttStreams.meeting && typeof sttStreams.meeting.reset === 'function') {
        sttStreams.meeting.reset();
      }
      partials.meeting = '';
      utteranceStart.meeting = null;
      status.set('pause', 'ask a question anytime');
      notify(
        opts.mic
          ? 'translate paused — mic back on for spoken questions; type or speak, or "start" again.'
          : 'translate paused — type questions, or "start" / "save".'
      );
    } else if (cmd === 'save') {
      if (!sessionDir) return notify('nothing to save yet — say or type "start" first.');
      ensureSessionState();
      const appName = path.basename(sessionDir).split('_').slice(1).join('_') || 'unknown-app';
      saveSession(sessionDir, {
        transcriptLogPath: transcript.logPath,
        summary: summarizer.summary,
        appName,
      });
      notify(`saved to ${sessionDir}`);
    } else if (cmd === 'load') {
      if (recording) return notify('type "stop" before loading another session.');
      const names = listSessions(DEFAULT_SESSIONS_DIR);
      if (!names.length) return notify('no saved sessions found.');
      awaitingLoadChoice = names;
      notify(
        'saved sessions:\n' +
          names.map((n, i) => `  ${i + 1}. ${n}`).join('\n') +
          '\nType a number to load, or anything else to cancel.'
      );
    }
  }

  function resolveLoadChoice(input, names) {
    const idx = Number(input) - 1;
    const name =
      Number.isInteger(idx) && names[idx] ? names[idx] : names.find((n) => n === input.trim());
    if (!name) return notify('load cancelled.');
    const { dir, transcriptText, summary, files } = loadSession(DEFAULT_SESSIONS_DIR, name);
    sessionDir = dir;
    ensureSessionState();
    const lines = transcriptText
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    lines.forEach((line, i) => retrieval.add({ line, text: line, elapsedMs: i }));
    files.forEach((f) =>
      chunkText(f.text).forEach((chunk, i) =>
        retrieval.add({ line: `[file: ${f.name} #${i + 1}] ${chunk}`, text: chunk, elapsedMs: 0 })
      )
    );
    if (summary) {
      summarizer.setSummary(
        summarizer.summary ? `${summarizer.summary}\n\n(From a loaded session:)\n${summary}` : summary
      );
    }
    notify(`loaded "${name}" (${lines.length} transcript line(s), ${files.length} file(s)).`);
  }

  function handleAddFile(rawPath) {
    if (!sessionDir) return notify('start a session first ("start") before adding a file.');
    const filePath = rawPath.replace(/^["']|["']$/g, '');
    let text;
    try {
      addFileToSession(sessionDir, filePath);
      text = fs.readFileSync(filePath, 'utf8');
    } catch (err) {
      return notify(`could not add file: ${err.message}`);
    }
    ensureSessionState();
    const chunks = chunkText(text);
    const name = path.basename(filePath);
    chunks.forEach((chunk, i) =>
      retrieval.add({ line: `[file: ${name} #${i + 1}] ${chunk}`, text: chunk, elapsedMs: 0 })
    );
    notify(`added "${name}" (${chunks.length} chunk(s)) — searchable for questions.`);
  }

  function resetSttBuffers({ sources = null } = {}) {
    for (const source of Object.keys(sttStreams)) {
      if (sources && !sources.includes(source)) continue;
      if (typeof sttStreams[source].reset === 'function') sttStreams[source].reset();
      partials[source] = '';
      utteranceStart[source] = null;
    }
  }

  /** True when loopback is the same device that plays TTS (default speakers). */
  function meetingLoopbackHearsTts() {
    // Separate virtual cable capture ⇒ TTS on Kraken won't re-enter meeting STT.
    if (opts.muteOriginal && opts.loopbackDevice) return false;
    if (opts.loopbackDevice && opts.speakersDevice) {
      const a = String(opts.loopbackDevice).toLowerCase();
      const b = String(opts.speakersDevice).toLowerCase();
      if (a && b && a !== b && !a.includes(b) && !b.includes(a)) return false;
    }
    return true;
  }

  function isAudioMuted(source) {
    // Mic: always mute while we speak (avoid command echo).
    if (source === 'you' && (speaking || Date.now() < muteUntil)) return true;
    // Meeting: only mute when loopback would hear our TTS / answer voice.
    if (source === 'meeting' && meetingLoopbackHearsTts()) {
      if (speaking || Date.now() < muteUntil) return true;
      if (answering && opts.tts) return true;
    }
    return false;
  }

  function normalizeEcho(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function isEchoOfLastSpoken(text) {
    const a = normalizeEcho(lastSpokenText);
    const b = normalizeEcho(text);
    if (!a || !b || a.length < 6) return false;
    if (a === b) return true;
    if (a.includes(b) || b.includes(a)) return true;
    const aw = new Set(a.split(' ').filter((w) => w.length > 1));
    const bw = b.split(' ').filter((w) => w.length > 1);
    if (!bw.length) return false;
    let hit = 0;
    for (const w of bw) if (aw.has(w)) hit++;
    return hit / bw.length >= 0.6;
  }

  async function speakAnswer(text, { allowClip = true, epoch = null } = {}) {
    if (!tts || !text) return;
    if (epoch != null && epoch !== speakEpoch) return;
    let spoken = String(text).trim();
    if (allowClip && spoken.length > 200) {
      const cut = spoken.slice(0, 200);
      const breakAt = Math.max(
        cut.lastIndexOf('. '),
        cut.lastIndexOf('! '),
        cut.lastIndexOf('? '),
        cut.lastIndexOf('। ')
      );
      spoken = (breakAt > 50 ? cut.slice(0, breakAt + 1) : cut).trim() + '…';
    }
    lastSpokenText = spoken;
    speaking = true;
    ttsStartedAt = Date.now();
    status.set('speak', spoken.slice(0, 48) + (spoken.length > 48 ? '…' : ''));
    // Never wipe meeting audio that was buffered during LLM translate.
    resetSttBuffers({ sources: meetingLoopbackHearsTts() ? ['you', 'meeting'] : ['you'] });
    try {
      await tts.speak(spoken);
    } catch (err) {
      notify(`TTS playback failed: ${err.message}`);
    } finally {
      speaking = false;
      resetSttBuffers({ sources: ['you'] });
      muteUntil = Date.now() + (meetingLoopbackHearsTts() ? 500 : 200);
      if (recording) status.set('listen', `→ ${opts.to}`);
      else status.stop();
    }
  }

  /** Fire-and-forget speak so the translate pump can start the next LLM job immediately. */
  function enqueueSpeak(text) {
    speakEpoch += 1;
    const epoch = speakEpoch;
    // Only interrupt if something is currently playing — don't kill the warm server every line.
    if (speaking && tts && typeof tts.stop === 'function') tts.stop();
    void speakAnswer(text, { epoch }).catch((err) => {
      log.error('TTS failed', err.message);
    });
  }

  async function handleQuestion(question) {
    if (answering) {
      printLine('[rcli-translate] still answering the previous question, please wait...');
      return;
    }
    answering = true;
    status.set('think', question.slice(0, 40) + (question.length > 40 ? '…' : ''));
    try {
      const recentLines = transcript ? transcript.lastMinutes(opts.minutes).map((s) => s.line) : [];
      const inWindow = new Set(recentLines);
      const retrieved = retrieval ? retrieval.topK(question, RETRIEVAL_TOP_K, inWindow) : [];
      const summary = summarizer ? summarizer.summary : '';

      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      process.stdout.write('>> ');
      let placeholderShown = false;
      let gotAnswerText = false;
      const clearPlaceholder = () => {
        if (!placeholderShown) return;
        readline.clearLine(process.stdout, 0);
        readline.cursorTo(process.stdout, 0);
        process.stdout.write('>> ');
        placeholderShown = false;
      };

      let answer = '';
      try {
        ({ answer } = await askQuestion(
          engine.llm,
          {
            summary,
            recentLines,
            retrievedLines: retrieved.map((r) => r.line),
            partials,
            question,
            disableThinking: engine.disableThinking,
            labels: opts.sourceLabels,
          },
          (text) => {
            clearPlaceholder();
            gotAnswerText = true;
            process.stdout.write(text);
          },
          () => {
            process.stdout.write('(thinking...)');
            placeholderShown = true;
          }
        ));
        clearPlaceholder();
        if (!gotAnswerText || !answer) {
          process.stdout.write(
            '[rcli-translate] the model used its whole budget reasoning without answering. Try a more specific question, or raise RCLI_XL8_ANSWER_TOKENS.'
          );
        }
      } catch (err) {
        clearPlaceholder();
        process.stdout.write(`\n[rcli-translate] answer failed: ${err.message}`);
      }
      process.stdout.write('\n');
      await speakAnswer(answer);
    } finally {
      answering = false;
    }
    while (deferredLines.length) process.stdout.write(deferredLines.shift() + '\n');
    rl.prompt(true);
  }

  async function pumpTranslateQueue() {
    if (translatePumpRunning) return;
    translatePumpRunning = true;
    translating = true;
    try {
      while (translateQueue.length) {
        const job = translateQueue.shift();
        try {
          await job();
        } catch (err) {
          notify(`translate failed: ${err.message}`);
        }
      }
    } finally {
      translating = false;
      translatePumpRunning = false;
      if (recording && !speaking && !answering) status.set('listen', `→ ${opts.to}`);
      while (deferredLines.length) process.stdout.write(deferredLines.shift() + '\n');
      rl.prompt(true);
    }
  }

  function enqueueMeetingTranslate({ text, lang, startedAt }) {
    const enqueuedAt = Date.now();
    const epoch = translateEpoch;
    log.event('asr.final', { text, lang, startedAt });
    status.set('translate', lang && lang !== 'und' ? `${lang} → ${opts.to}` : `→ ${opts.to}`);

    // Stay live: never pile up old speech. Keep at most one waiting job.
    // Also bump a job token so an in-flight LLM can be abandoned after it returns.
    if (translateQueue.length > 0) {
      const dropped = translateQueue.length;
      translateQueue.length = 0;
      log.warn('dropped stale translate jobs to stay live', { n: dropped });
    }
    const jobToken = ++translateJobToken;

    translateQueue.push(async () => {
      if (epoch !== translateEpoch || !recording) {
        log.info('abandon translate job (stopped/superseded)');
        return;
      }
      const waited = Date.now() - enqueuedAt;
      const stale = waited > 10000;
      ensureSessionState();
      const recentContext = transcript
        ? transcript
            .lastMinutes(2)
            .slice(-2)
            .map((s) => s.text || s.line)
            .join(' | ')
            .slice(0, 220)
        : '';
      const asrText = text.length > 360 ? text.slice(0, 360).trim() + '…' : text;
      log.event('translate.start', { text: asrText, lang, to: opts.to, waitedMs: waited, stale });
      let result;
      try {
        result = await translateUtterance(engine.llm, {
          text: asrText,
          to: opts.to,
          srcLangHint: lang,
          recentContext,
          disableThinking: engine.disableThinking,
        });
      } catch (err) {
        log.error('translate failed', err.message);
        notify(`translate failed: ${err.message}`);
        return;
      }
      // A newer utterance started translating — drop this result (except we already
      // paid for LLM; still skip caption/TTS to stay live).
      if (jobToken !== translateJobToken || epoch !== translateEpoch || !recording) {
        log.info('discard superseded translate result');
        return;
      }

      log.event('translate.done', {
        ms: Date.now() - enqueuedAt,
        fast: !!result.fast,
        lang: result.lang,
        targetOk: !!result.targetOk,
        translation: result.translation,
        repaired: result.repaired,
      });

      const display =
        result.targetOk && result.translation ? result.translation : result.repaired || text;
      if (!display) {
        log.warn('translate returned empty');
        return;
      }

      const meta = {
        raw: text,
        repaired: result.repaired || text,
        translation: result.targetOk ? result.translation : '',
        srcLang: result.lang || lang || 'und',
        tgtLang: result.targetOk ? opts.to : result.lang || lang || 'und',
      };
      const seg = transcript.add(display, 'meeting', startedAt, meta);
      retrieval.add(seg, { defer: true });
      summarizer.addSegment(seg);
      // Don't steal the LLM lock for summaries while live-translating.
      if (!recording) summarizer.maybeUpdate();
      const elapsed = Date.now() - enqueuedAt;
      const canSpeak = !!(
        tts &&
        result.targetOk &&
        result.translation &&
        !stale &&
        translateQueue.length === 0 &&
        epoch === translateEpoch &&
        recording
      );
      const pretty = formatFinalLine({
        line: seg.line,
        ms: elapsed,
        spoken: canSpeak,
      });
      if (answering) deferredLines.push(pretty);
      else {
        printLine(pretty);
        // Show source ASR under the translation when it differs (debug + trust).
        if (result.targetOk && result.repaired && result.repaired !== display) {
          printLine(paint(c.dim, `  ↳ ${result.repaired.slice(0, 140)}${result.repaired.length > 140 ? '…' : ''}`));
        }
      }

      if (!canSpeak) {
        if (!result.targetOk) {
          log.warn('skip TTS — translation not in target script/language', {
            to: opts.to,
            sample: String(result.translation || display).slice(0, 80),
          });
        } else {
          log.info('skip TTS to stay live', { stale, queued: translateQueue.length });
        }
        return;
      }
      // Do NOT await TTS — next ASR→LLM translate must start immediately.
      enqueueSpeak(result.translation);
    });
    void pumpTranslateQueue();
  }

  function wireStream(sttStream, source) {
    sttStream.on('error', (err) => {
      log.error(`${source} stt`, err && err.message ? err.message : err);
      notify(`${source} transcription error: ${err && err.message ? err.message : err}`);
    });

    sttStream.on('speech-start', () => {
      if (!recording && source === 'meeting') return;
      log.event('speech.start', { source });
      // Partials are off for meeting — stamp start here so captions aren't "now".
      if (utteranceStart[source] == null) {
        utteranceStart[source] = transcript ? transcript.elapsedNow() : Date.now();
      }
      if (source === 'meeting' && recording && !answering && !translating) {
        status.set('listen', 'hearing…');
      }
    });

    sttStream.on('partial', (text, lang) => {
      if (isAudioMuted(source)) return;
      if (!recording && source === 'meeting') return;
      if (source === 'meeting' && isEchoOfLastSpoken(text)) return;
      if (utteranceStart[source] == null) {
        utteranceStart[source] = transcript ? transcript.elapsedNow() : Date.now();
      }
      partials[source] = text;
      log.event('asr.partial', { source, lang, text });
      if (answering || translating) return;
      status.stop();
      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      let tag = opts.sourceLabels[source] || source;
      if (source === 'meeting' && lang && lang !== 'und') tag = `${tag}/${lang}`;
      if (!recording && source === 'you') tag = `${opts.sourceLabels.you} → chat`;
      const row = formatPartialLine({
        tag,
        text,
        mode: source === 'you' ? 'you' : 'hear',
      });
      process.stdout.write(fitOneRow(row));
    });

    sttStream.on('final', (payload) => {
      const { text, lang, rawText, msAudio, wallStartedAt } = normalizeFinalPayload(payload);
      const startedAt = utteranceStart[source] ?? (transcript ? transcript.elapsedNow() : Date.now());
      utteranceStart[source] = null;
      partials[source] = '';
      const mutedNow = isAudioMuted(source);
      // Keep finals that began BEFORE TTS — they are real Meet speech, not echo.
      const startedBeforeTts =
        Number.isFinite(wallStartedAt) && ttsStartedAt > 0 && wallStartedAt < ttsStartedAt - 50;
      const dropForMute = mutedNow && !(source === 'meeting' && startedBeforeTts);
      log.event('asr.final.raw', {
        source,
        lang,
        text,
        rawText,
        scrubbed: !!(rawText && !text),
        msAudio,
        muted: mutedNow,
        keptDespiteMute: mutedNow && !dropForMute,
        recording,
      });
      if (dropForMute) return;
      if (!recording && source === 'meeting') return;
      if (source === 'meeting' && text && isEchoOfLastSpoken(text)) {
        log.info('ignored meeting echo of TTS');
        return;
      }
      if (!text) {
        if (source === 'meeting' && recording) {
          log.warn('empty ASR final for meeting — no translate', { rawText, lang, msAudio });
          status.set('listen', `→ ${opts.to}`);
        }
        return;
      }

      if (source === 'you') {
        const spokenCmd = parseCommand(text, { allowStop: false });
        if (spokenCmd) {
          notify(`heard command "${spokenCmd}" (from: "${text}")`);
          void handleCommand(spokenCmd);
          return;
        }
      }

      if (recording && source === 'meeting') {
        lastMeetingAsrAt = Date.now();
        enqueueMeetingTranslate({ text, lang, startedAt });
        return;
      }

      if (recording && source === 'you') {
        ensureSessionState();
        const seg = transcript.add(text, source, startedAt);
        retrieval.add(seg, { defer: true });
        summarizer.addSegment(seg);
        // Live translate owns the LLM — don't summarize mid-call.
        if (!recording) summarizer.maybeUpdate();
        const pretty = formatFinalLine({ line: seg.line, spoken: false });
        if (answering || translating) deferredLines.push(pretty);
        else printLine(pretty);
        return;
      }

      if (!recording && source === 'you') {
        printLine(`[${opts.sourceLabels.you}] ${text}`);
        void handleQuestion(text);
      }
    });
  }

  for (const source of SOURCES) {
    if (source === 'you' && !opts.mic) continue;
    const streamOpts =
      source === 'meeting'
        ? {
            language: opts.from || 'auto',
            // Partials steal the only GPU Whisper worker from finals → late voice.
            partials: false,
            priority: 10, // meeting beats mic on the shared decode queue
            prompt:
              opts.from && opts.from !== 'auto'
                ? `Meeting speech in ${opts.from}. Transcribe accurately.`
                : 'Casual Discord/Meet chat. Indian English accents OK. Transcribe English as English (Latin), Hindi as Hindi. Keep slang, names, Discord, PR, standup.',
          }
        : { language: 'en', prompt: MIC_PROMPT, partials: false, priority: 0 };
    const sttStream = stt.createStream(streamOpts);
    wireStream(sttStream, source);
    sttStreams[source] = sttStream;
    captures[source] = startCapture(
      source === 'meeting' ? 'loopback' : 'mic',
      (samples) => {
        if (isAudioMuted(source)) return;
        if (!recording && source === 'meeting') return;
        // While live-translating, keep the only Whisper worker on Meet audio.
        // Mic STT resumes automatically on "stop" for spoken Q&A.
        if (recording && source === 'you') return;
        try {
          sttStream.feed(samples);
        } catch (err) {
          notify(`${source} transcription error: ${err.message}`);
        }
      },
      (message) => {
        notify(message);
        notify(`${source} captions have stopped; other sources keep working.`);
      },
      source === 'you'
        ? { device: opts.micName, gain: opts.micGain }
        : { device: opts.loopbackDevice }
    );
  }

  rl.on('line', (line) => void handleLine(line));

  async function handleLine(rawLine) {
    const line = rawLine.trim();
    if (!line) return rl.prompt();
    if (line === '/quit' || line === '/exit') return void shutdown();

    if (awaitingLoadChoice) {
      const names = awaitingLoadChoice;
      awaitingLoadChoice = null;
      return resolveLoadChoice(line, names);
    }

    const addMatch = /^add\s+(.+)$/i.exec(line);
    if (addMatch) return handleAddFile(addMatch[1].trim());

    const cmd = parseCommand(line, { allowStop: true });
    if (cmd) return void handleCommand(cmd);

    void handleQuestion(line);
  }

  let shuttingDown = false;
  async function shutdown() {
    if (shuttingDown) return;
    shuttingDown = true;
    clearInterval(heartbeat);
    status.stop();
    speakEpoch += 1;
    translateEpoch += 1;
    translateQueue.length = 0;
    while (deferredLines.length) process.stdout.write(deferredLines.shift() + '\n');
    process.stdout.write('\n' + paint(c.dim, 'rcli-translate · bye') + '\n');
    log.info('shutting down');
    log.close();
    for (const source of Object.keys(captures)) captures[source].stop();
    for (const source of Object.keys(sttStreams)) sttStreams[source].close();
    stt.close();
    if (tts) tts.close();
    if (transcript) await transcript.close();
    engine.shutdown();
    rl.close();
    process.exit(0);
  }

  rl.on('SIGINT', () => void shutdown());
  process.on('SIGINT', () => void shutdown());

  process.stdout.write(
    banner({ to: opts.to, other: opts.sourceLabels.meeting, tts: opts.tts })
  );
  console.log(
    paint(c.dim, `  window ${opts.minutes}m · context ${CONTEXT_TOKEN_BUDGET} tok · STT ${STT_ENGINE}`)
  );
  if (opts.autostart) {
    beginSession('autostart');
  } else {
    notify('Type or say "start" to begin live translation.');
  }
  notify('Type "stop" to pause and ask questions. /quit to exit.');
  rl.prompt();
}

main().catch((err) => {
  console.log(`[rcli-translate] fatal error: ${err && err.message ? err.message : err}`);
  process.exit(1);
});
