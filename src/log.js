// Append-only file logger so debugging doesn't require pasting the terminal.
const fs = require('fs');
const path = require('path');

const LOG_DIR = path.join(__dirname, '..', 'logs');

function stamp() {
  return new Date().toISOString();
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

/**
 * @param {{ sessionDir?: string, alsoConsole?: boolean }} opts
 */
function createLogger(opts = {}) {
  ensureDir(LOG_DIR);
  const latestPath = path.join(LOG_DIR, 'latest.log');
  const runPath = path.join(LOG_DIR, `run-${Date.now()}.log`);
  const streams = [
    fs.createWriteStream(latestPath, { flags: 'w' }),
    fs.createWriteStream(runPath, { flags: 'a' }),
  ];
  let sessionStream = null;

  function writeAll(line) {
    for (const s of streams) {
      try {
        s.write(line);
      } catch {
        /* ignore */
      }
    }
    if (sessionStream) {
      try {
        sessionStream.write(line);
      } catch {
        /* ignore */
      }
    }
  }

  const api = {
    latestPath,
    runPath,
    sessionPath: null,

    attachSession(sessionDir) {
      if (!sessionDir) return;
      try {
        ensureDir(sessionDir);
        const p = path.join(sessionDir, 'debug.log');
        sessionStream = fs.createWriteStream(p, { flags: 'a' });
        api.sessionPath = p;
        writeAll(`${stamp()} [log] session debug -> ${p}\n`);
      } catch (err) {
        writeAll(`${stamp()} [log] could not attach session log: ${err.message}\n`);
      }
    },

    info(msg, extra) {
      const line = `${stamp()} [info] ${msg}${extra != null ? ' ' + safe(extra) : ''}\n`;
      writeAll(line);
    },

    warn(msg, extra) {
      const line = `${stamp()} [warn] ${msg}${extra != null ? ' ' + safe(extra) : ''}\n`;
      writeAll(line);
    },

    error(msg, extra) {
      const line = `${stamp()} [error] ${msg}${extra != null ? ' ' + safe(extra) : ''}\n`;
      writeAll(line);
    },

    event(kind, data) {
      const line = `${stamp()} [${kind}] ${safe(data)}\n`;
      writeAll(line);
    },

    close() {
      for (const s of streams) {
        try {
          s.end();
        } catch {
          /* ignore */
        }
      }
      if (sessionStream) {
        try {
          sessionStream.end();
        } catch {
          /* ignore */
        }
      }
    },
  };

  writeAll(`${stamp()} [log] rcli-translate starting\n`);
  return api;
}

function safe(v) {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

module.exports = { createLogger, LOG_DIR };
