// Terminal chrome for rcli-translate — colors, status line, banner.
// Keeps ANSI optional: disabled when stdout isn't a TTY or NO_COLOR is set.

const ENABLE =
  !!process.stdout.isTTY &&
  !process.env.NO_COLOR &&
  !/^(0|false|off)$/i.test(process.env.FORCE_COLOR || '1');

const c = {
  reset: ENABLE ? '\x1b[0m' : '',
  dim: ENABLE ? '\x1b[2m' : '',
  bold: ENABLE ? '\x1b[1m' : '',
  cyan: ENABLE ? '\x1b[36m' : '',
  teal: ENABLE ? '\x1b[38;5;44m' : '',
  amber: ENABLE ? '\x1b[38;5;214m' : '',
  green: ENABLE ? '\x1b[32m' : '',
  magenta: ENABLE ? '\x1b[35m' : '',
  red: ENABLE ? '\x1b[31m' : '',
  white: ENABLE ? '\x1b[97m' : '',
};

const SPIN = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

function paint(color, text) {
  return `${color}${text}${c.reset}`;
}

function banner({ to = 'en', other = 'other', tts = true } = {}) {
  const title = paint(c.teal + c.bold, 'rcli-translate');
  const tag = paint(c.amber, 'live');
  const lines = [
    '',
    `  ${title}  ${tag}`,
    paint(c.dim, '  auto-detect them → translate → captions' + (tts ? ' + voice' : '')),
    paint(c.dim, `  into ${to}  ·  other [${other}]  ·  headphones on`),
    '',
  ];
  return lines.join('\n');
}

function formatPartialLine({ tag, text, mode = 'hear' }) {
  const glyph = mode === 'you' ? '◆' : '◇';
  const color = mode === 'you' ? c.magenta : c.cyan;
  return paint(c.dim, `${glyph} `) + paint(color, `[${tag}]`) + ' ' + paint(c.white, text);
}

function formatFinalLine({ line, ms, spoken }) {
  let out = paint(c.green + c.bold, '✓ ') + line;
  const bits = [];
  if (Number.isFinite(ms) && ms > 0) bits.push(`${(ms / 1000).toFixed(1)}s`);
  if (spoken) bits.push('spoken');
  if (bits.length) out += paint(c.dim, `  · ${bits.join(' · ')}`);
  return out;
}

function formatStatus({ state, detail = '' }) {
  const map = {
    listen: { color: c.cyan, label: 'listening' },
    translate: { color: c.amber, label: 'translating' },
    speak: { color: c.green, label: 'speaking' },
    think: { color: c.magenta, label: 'thinking' },
    pause: { color: c.dim, label: 'paused' },
    ready: { color: c.teal, label: 'ready' },
  };
  const m = map[state] || map.ready;
  const tip = detail ? paint(c.dim, `  ${detail}`) : '';
  return paint(m.color, `● ${m.label}`) + tip;
}

/**
 * Live status line that repaints in place with a spinner.
 * Call .set(state, detail) and .stop() when done.
 */
function createStatusLine({ write = (s) => process.stdout.write(s), columns = () => process.stdout.columns || 80 } = {}) {
  let timer = null;
  let frame = 0;
  let state = 'ready';
  let detail = '';
  let visible = false;

  function row() {
    const spin = SPIN[frame % SPIN.length];
    const map = {
      listen: { color: c.cyan, label: 'listening' },
      translate: { color: c.amber, label: 'translating' },
      speak: { color: c.green, label: 'speaking' },
      think: { color: c.magenta, label: 'thinking' },
      pause: { color: c.dim, label: 'paused' },
      ready: { color: c.teal, label: 'ready' },
    };
    const m = map[state] || map.ready;
    const tip = detail ? `  ${detail}` : '';
    const raw = `${spin} ${m.label}${tip}`;
    const width = Math.max(20, columns() - 1);
    const clipped = raw.length <= width ? raw : raw.slice(0, width - 1) + '…';
    return paint(m.color, clipped);
  }

  function paintLine() {
    if (!process.stdout.isTTY) return;
    write('\r\x1b[K' + row());
    visible = true;
  }

  function set(nextState, nextDetail = '') {
    state = nextState;
    detail = nextDetail;
    if (!timer && process.stdout.isTTY) {
      timer = setInterval(() => {
        frame++;
        paintLine();
      }, 80);
      if (timer.unref) timer.unref();
    }
    paintLine();
  }

  function stop() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (visible && process.stdout.isTTY) {
      write('\r\x1b[K');
      visible = false;
    }
  }

  return { set, stop, get state() { return state; } };
}

function fitOneRow(text, columns = process.stdout.columns || 80) {
  const width = Math.max(20, columns - 1);
  // Strip ANSI for length check
  const plain = String(text).replace(/\x1b\[[0-9;]*m/g, '');
  if (plain.length <= width) return text;
  return '…' + plain.slice(-(width - 1));
}

module.exports = {
  c,
  ENABLE,
  paint,
  banner,
  formatPartialLine,
  formatFinalLine,
  formatStatus,
  createStatusLine,
  fitOneRow,
};
