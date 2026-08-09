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
  --to <lang>          Target language code for translations (default: en)
  --minutes <n>        Recency window for Q&A (default: 20)
  --llm <id|path>      RunAnywhere LLM catalog id or local GGUF path
  --embedder <id>      RunAnywhere embedder catalog id
  --no-mic             Don't capture your microphone
  --mic <name>         Substring to pick a mic (e.g. "Razer")
  --mic-gain <n>       Software mic gain (default 1)
  --no-tts             Captions only — don't speak translations or answers
  --other <name>       Label for the other person (default: other)
  --no-autostart       Wait for "start" instead of translating immediately
  -h, --help           Show this help

In-session:
  start / record       Begin (or resume) live translation + session capture
  stop                 (typed only) Pause translation — ask questions / voice chat
  save / load / add   Session persistence (same as rcli-meet)
  anything else        Ask a question over the bilingual transcript
  /quit                Exit

Use headphones so your mic does not pick up translated TTS or Meet speakers.
Latency after they finish speaking is typically ~1–3s (VAD + Whisper + LLM).

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
    other: env('OTHER', 'other'),
    to: (env('TO', 'en') || 'en').toLowerCase(),
    autostart: true,
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
    else if (arg === '--no-tts') opts.tts = false;
    else if (arg === '--no-autostart') opts.autostart = false;
    else throw new UsageError(`unknown option "${arg}"`);
  }
  opts.sourceLabels = resolveSourceLabels({ meeting: opts.other, you: 'you' });
  return opts;
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
    return { text: String(payload.text || '').trim(), lang: String(payload.lang || 'und').toLowerCase() };
  }
  return { text: String(payload || '').trim(), lang: 'und' };
}

async function main() {
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
  if (opts.help) {
    console.log(USAGE);
    return;
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

  const tts = opts.tts ? createTTS() : null;
  console.log(
    opts.tts
      ? '[rcli-translate] TTS ready — translations (and Q&A answers) will be spoken.'
      : '[rcli-translate] TTS disabled (--no-tts).'
  );
  if (opts.mic) {
    console.log(
      '[rcli-translate] mic capture enabled' +
        (opts.micName ? ` (device filter: "${opts.micName}")` : ' (Windows default input)') +
        (opts.micGain !== 1 ? `, gain×${opts.micGain}` : '') +
        '.'
    );
  }
  console.log(
    `[rcli-translate] translating meeting → ${opts.to}; other labeled [${opts.sourceLabels.meeting}].`
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
  let lastSpokenText = '';
  let recording = false; // live-translate / capture active
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
    status.set('listen', `→ ${opts.to}`);
    notify(
      `live translate ON (${reason}; window: "${appName}"; → ${opts.to}). Session: ${sessionDir}`
    );
  }

  async function handleCommand(cmd) {
    if (cmd === 'start') {
      if (recording) return notify('already translating.');
      beginSession('start');
    } else if (cmd === 'stop') {
      if (!recording) return notify('not currently translating.');
      recording = false;
      if (sttStreams.meeting && typeof sttStreams.meeting.reset === 'function') {
        sttStreams.meeting.reset();
      }
      partials.meeting = '';
      utteranceStart.meeting = null;
      status.set('pause', 'ask a question anytime');
      notify(
        opts.mic
          ? 'translate paused — ask typed/spoken questions about the transcript, or "start" again.'
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

  function resetSttBuffers() {
    for (const source of Object.keys(sttStreams)) {
      if (typeof sttStreams[source].reset === 'function') sttStreams[source].reset();
      partials[source] = '';
      utteranceStart[source] = null;
    }
  }

  function isAudioMuted(source) {
    if (speaking || Date.now() < muteUntil) return true;
    if (source === 'meeting' && (answering || translating) && opts.tts) return true;
    return false;
  }

  function normalizeEcho(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function isEchoOfLastSpoken(text) {
    const a = normalizeEcho(lastSpokenText);
    const b = normalizeEcho(text);
    if (!a || !b || a.length < 8) return false;
    if (a === b) return true;
    if (a.includes(b) || b.includes(a)) return true;
    const aw = new Set(a.split(' ').filter((w) => w.length > 2));
    const bw = b.split(' ').filter((w) => w.length > 2);
    if (!bw.length) return false;
    let hit = 0;
    for (const w of bw) if (aw.has(w)) hit++;
    return hit / bw.length >= 0.6;
  }

  async function speakAnswer(text) {
    if (!tts || !text) return;
    lastSpokenText = text;
    speaking = true;
    status.set('speak', text.slice(0, 48) + (text.length > 48 ? '…' : ''));
    resetSttBuffers();
    try {
      await tts.speak(text);
    } catch (err) {
      notify(`TTS playback failed: ${err.message}`);
    } finally {
      speaking = false;
      resetSttBuffers();
      muteUntil = Date.now() + 1800;
      if (recording) status.set('listen', `→ ${opts.to}`);
      else status.stop();
    }
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
    const t0 = Date.now();
    status.set('translate', lang && lang !== 'und' ? `${lang} → ${opts.to}` : `→ ${opts.to}`);
    translateQueue.push(async () => {
      ensureSessionState();
      const recentContext = transcript
        ? transcript
            .lastMinutes(3)
            .slice(-6)
            .map((s) => s.line)
            .join('\n')
        : '';
      const result = await translateUtterance(engine.llm, {
        text,
        to: opts.to,
        srcLangHint: lang,
        recentContext,
        disableThinking: engine.disableThinking,
      });
      if (!result.translation && !result.repaired) return;

      const meta = {
        raw: text,
        repaired: result.repaired || text,
        translation: result.translation || result.repaired || text,
        srcLang: result.lang || lang || 'und',
        tgtLang: opts.to,
      };
      const seg = transcript.add(meta.translation, 'meeting', startedAt, meta);
      retrieval.add(seg);
      summarizer.addSegment(seg);
      summarizer.maybeUpdate();
      const elapsed = Date.now() - t0;
      const pretty = formatFinalLine({
        line: seg.line,
        ms: elapsed,
        spoken: !!(tts && meta.translation),
      });
      if (answering) deferredLines.push(pretty);
      else printLine(pretty);
      await speakAnswer(meta.translation);
    });
    void pumpTranslateQueue();
  }

  function wireStream(sttStream, source) {
    sttStream.on('error', (err) => {
      notify(`${source} transcription error: ${err && err.message ? err.message : err}`);
    });

    sttStream.on('partial', (text, lang) => {
      if (isAudioMuted(source)) return;
      if (!recording && source === 'meeting') return;
      if (source === 'meeting' && isEchoOfLastSpoken(text)) return;
      if (utteranceStart[source] == null) {
        utteranceStart[source] = transcript ? transcript.elapsedNow() : Date.now();
      }
      partials[source] = text;
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
      const { text, lang } = normalizeFinalPayload(payload);
      const startedAt = utteranceStart[source] ?? (transcript ? transcript.elapsedNow() : Date.now());
      utteranceStart[source] = null;
      partials[source] = '';
      if (isAudioMuted(source)) return;
      if (!recording && source === 'meeting') return;
      if (source === 'meeting' && isEchoOfLastSpoken(text)) {
        notify('(ignored meeting echo of my own voice)');
        return;
      }
      if (!text) return;

      if (source === 'you') {
        const spokenCmd = parseCommand(text, { allowStop: false });
        if (spokenCmd) {
          notify(`heard command "${spokenCmd}" (from: "${text}")`);
          void handleCommand(spokenCmd);
          return;
        }
      }

      if (recording && source === 'meeting') {
        enqueueMeetingTranslate({ text, lang, startedAt });
        return;
      }

      if (recording && source === 'you') {
        ensureSessionState();
        const seg = transcript.add(text, source, startedAt);
        retrieval.add(seg);
        summarizer.addSegment(seg);
        summarizer.maybeUpdate();
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
        ? { language: 'auto', prompt: 'Live multilingual meeting speech. Transcribe in the spoken language.' }
        : { language: 'en', prompt: MIC_PROMPT };
    const sttStream = stt.createStream(streamOpts);
    wireStream(sttStream, source);
    sttStreams[source] = sttStream;
    captures[source] = startCapture(
      source === 'meeting' ? 'loopback' : 'mic',
      (samples) => {
        if (isAudioMuted(source)) return;
        if (!recording && source === 'meeting') return;
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
      source === 'you' ? { device: opts.micName, gain: opts.micGain } : {}
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
    status.stop();
    while (deferredLines.length) process.stdout.write(deferredLines.shift() + '\n');
    process.stdout.write('\n' + paint(c.dim, 'rcli-translate · bye') + '\n');
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
