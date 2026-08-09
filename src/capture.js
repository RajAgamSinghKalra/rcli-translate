// Spawns capture_audio.py (loopback or mic) and turns its raw float32 PCM
// stdout stream into Float32Array chunks for the STT engine.
//
// Critical: always drain the child's stdout promptly. Heavy STT work is
// deferred via setImmediate so a busy Whisper decode can't stall the pipe
// reader -- that backpressure is what triggers WASAPI "data discontinuity".
const fs = require('fs');
const path = require('path');
const { spawnPython } = require('./python');
const SCRIPT = path.join(__dirname, '..', 'capture_audio.py');
const BYTES_PER_SAMPLE = 4; // float32

/** Hide the known-noisy soundcard discontinuity warning from the live UI. */
function filterCaptureStderr(text) {
  return text
    .split(/\r?\n/)
    .filter((line) => {
      if (!line) return false;
      if (/data discontinuity in recording/i.test(line)) return false;
      // Mic silence is noisy and unrelated to meeting loopback translate.
      if (/WARNING: microphone is returning silence/i.test(line)) return false;
      if (/Check hardware mute/i.test(line)) return false;
      if (/List devices:/i.test(line) && /capture_audio/i.test(line)) return false;
      if (/Razer Kraken mute/i.test(line)) return false;
      if (/Razer Synapse/i.test(line)) return false;
      if (/Windows Settings/i.test(line)) return false;
      if (/mic-level rms=/i.test(line)) return false;
      return true;
    })
    .join('\n');
}

/**
 * @param source {'loopback'|'mic'}
 * @param onSamples {(samples: Float32Array) => void}
 * @param onFatal {(message: string) => void}
 * @param opts {{ device?: string, gain?: number }}
 */
function startCapture(source, onSamples, onFatal = () => {}, opts = {}) {
  if (source !== 'loopback' && source !== 'mic') {
    throw new Error(`startCapture: source must be "loopback" or "mic", got "${source}"`);
  }
  if (!fs.existsSync(SCRIPT)) {
    onFatal(`capture helper not found at ${SCRIPT}`);
    return { stop() {} };
  }

  const args = [SCRIPT, '--source', source];
  const device =
    opts.device ||
    (source === 'mic'
      ? process.env.RCLI_XL8_MIC || process.env.RCLI_MEET_MIC || ''
      : process.env.RCLI_XL8_LOOPBACK || process.env.RCLI_MEET_LOOPBACK || '');
  if (device) {
    args.push('--device', device);
  }
  const gain =
    opts.gain ?? Number(process.env.RCLI_XL8_MIC_GAIN || process.env.RCLI_MEET_MIC_GAIN || '1');
  if (source === 'mic' && Number.isFinite(gain) && gain > 0 && gain !== 1) {
    args.push('--gain', String(gain));
  }

  const proc = spawnPython(args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let pending = Buffer.alloc(0);
  let stopped = false;
  let stderrTail = '';
  const workQueue = [];
  let drainScheduled = false;

  function drainWork() {
    drainScheduled = false;
    const batch = Math.min(workQueue.length, 8);
    for (let i = 0; i < batch; i++) {
      try {
        onSamples(workQueue.shift());
      } catch (err) {
        onFatal(`${source} sample handler error: ${err.message}`);
      }
    }
    if (workQueue.length) {
      drainScheduled = true;
      setImmediate(drainWork);
    }
  }

  proc.on('error', (err) => {
    if (stopped) return;
    const hint =
      err.code === 'ENOENT'
        ? `\n  Could not run Python. Set RCLI_XL8_PYTHON to your real python.exe` +
          `\n  (on Windows, a bare "python" often resolves to the Store alias stub).`
        : '';
    onFatal(`could not start ${source} capture: ${err.message}${hint}`);
  });

  proc.stdout.on('data', (chunk) => {
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;

    const usableLen = pending.length - (pending.length % BYTES_PER_SAMPLE);
    if (usableLen === 0) return;

    const sampleCount = usableLen / BYTES_PER_SAMPLE;
    const floats = new Float32Array(sampleCount);
    Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength).set(
      pending.subarray(0, usableLen)
    );
    pending =
      usableLen === pending.length ? Buffer.alloc(0) : Buffer.from(pending.subarray(usableLen));

    workQueue.push(floats);
    while (workQueue.length > 40) workQueue.shift();

    if (!drainScheduled) {
      drainScheduled = true;
      setImmediate(drainWork);
    }
  });

  proc.stderr.on('data', (chunk) => {
    const text = String(chunk);
    stderrTail = (stderrTail + text).slice(-4000);
    const visible = filterCaptureStderr(text);
    if (visible.trim()) process.stdout.write(`[capture:${source}] ${visible}\n`);
  });

  proc.on('exit', (code, signal) => {
    if (stopped) return;
    if (code !== null && code !== 0) {
      const detail = stderrTail.trim() ? `\n${stderrTail.trim()}` : '';
      onFatal(`${source} capture stopped unexpectedly (exit code ${code})${detail}`);
    } else if (signal) {
      onFatal(`${source} capture stopped unexpectedly (killed by ${signal})`);
    } else {
      onFatal(`${source} capture stopped unexpectedly (helper exited)`);
    }
  });

  return {
    stop() {
      if (stopped) return;
      stopped = true;
      workQueue.length = 0;
      proc.kill();
    },
  };
}

module.exports = { startCapture, filterCaptureStderr };
