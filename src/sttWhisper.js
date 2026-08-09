// Offline Whisper STT via native sherpa-onnx-node (NOT the WASM package).
// Kept as a CPU fallback when Vulkan whisper.cpp isn't installed. Prefer
// RCLI_MEET_STT_ENGINE=vulkan (default) for GPU on AMD/NVIDIA.
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const sherpa_onnx = require('./sherpa');
const { createEnergyVad, SAMPLE_RATE } = require('./vadEnergy');

const MODEL_FILES = {
  encoder: 'small.en-encoder.int8.onnx',
  decoder: 'small.en-decoder.int8.onnx',
  tokens: 'small.en-tokens.txt',
};

const MODEL_URL =
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2';

function assertModelPresent(modelDir) {
  const missing = Object.values(MODEL_FILES).filter((f) => !fs.existsSync(path.join(modelDir, f)));
  if (missing.length === 0) return;
  throw new Error(
    `Whisper STT model files missing from:\n  ${modelDir}\n` +
      missing.map((f) => `  - ${f}`).join('\n') +
      `\n\nDownload with:\n` +
      `  curl -L -o models/whisper.tar.bz2 ${MODEL_URL}\n` +
      `  tar -xjf models/whisper.tar.bz2 -C models/\n` +
      `\nOr (recommended) use GPU Whisper instead:\n` +
      `  powershell -ExecutionPolicy Bypass -File scripts/setup-stt-gpu.ps1`
  );
}

function createSTTEngine(modelDir) {
  assertModelPresent(modelDir);

  const recognizer = new sherpa_onnx.OfflineRecognizer({
    modelConfig: {
      whisper: {
        encoder: path.join(modelDir, MODEL_FILES.encoder),
        decoder: path.join(modelDir, MODEL_FILES.decoder),
        language: 'en',
        task: 'transcribe',
      },
      tokens: path.join(modelDir, MODEL_FILES.tokens),
      numThreads: Math.max(2, Math.min(6, require('os').cpus().length - 1 || 2)),
      provider: 'cpu',
      debug: 0,
    },
  });

  let engineClosed = false;
  const openStreams = new Set();
  // Serialize offline decodes so dual streams don't thrash the CPU recognizer.
  let decodeChain = Promise.resolve();

  function decodeAsync(samples) {
    const run = decodeChain.then(async () => {
      if (engineClosed) return '';
      const stream = recognizer.createStream();
      stream.acceptWaveform({ samples, sampleRate: SAMPLE_RATE });
      const result = await recognizer.decodeAsync(stream);
      return (result.text || '').trim();
    });
    decodeChain = run.then(
      () => {},
      () => {}
    );
    return run;
  }

  function createStream() {
    if (engineClosed) throw new Error('createStream: STT engine is already closed');

    const emitter = new EventEmitter();
    let streamClosed = false;

    const vad = createEnergyVad({
      onSpeechStart: () => emitter.emit('partial', '(listening)'),
      onUtterance: (samples) => {
        decodeAsync(samples)
          .then((text) => {
            if (!streamClosed && !engineClosed && text) emitter.emit('final', text);
          })
          .catch((err) => {
            if (!streamClosed && !engineClosed) emitter.emit('error', err);
          });
      },
    });

    emitter.sampleRate = SAMPLE_RATE;
    emitter.feed = function feed(samples) {
      if (streamClosed || engineClosed) return;
      vad.feed(samples);
    };
    emitter.close = function close() {
      if (streamClosed) return;
      streamClosed = true;
      openStreams.delete(emitter);
      vad.flush();
    };
    emitter.reset = function reset() {
      vad.clear();
    };

    openStreams.add(emitter);
    return emitter;
  }

  return {
    createStream,
    close() {
      if (engineClosed) return;
      engineClosed = true;
      for (const s of Array.from(openStreams)) s.close();
    },
  };
}

module.exports = {
  createSTTEngine,
  assertModelPresent,
  createEnergyVad,
  SAMPLE_RATE,
  MODEL_FILES,
};
