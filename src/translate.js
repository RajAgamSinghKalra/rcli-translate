// Live repair + translate: one LLM call turns raw multilingual ASR into a
// cleaned source string and a translation into the user's target language.
const { serialize, NO_THINK_DIRECTIVE, TRANSCRIPTION_CAVEAT } = require('./llm');

const TRANSLATE_MAX_TOKENS =
  Number(process.env.RCLI_XL8_TRANSLATE_TOKENS || process.env.RCLI_MEET_TRANSLATE_TOKENS) || 140;
const TRANSLATE_TEMPERATURE = 0.1;
const ALWAYS_LLM = /^(1|on|true|yes)$/i.test(process.env.RCLI_XL8_ALWAYS_LLM || '');

const LANG_NAMES = {
  en: 'English',
  hi: 'Hindi (Devanagari हिन्दी)',
  es: 'Spanish',
  fr: 'French',
  de: 'German',
  pt: 'Portuguese',
  ja: 'Japanese',
  ko: 'Korean',
  zh: 'Chinese',
  ar: 'Arabic',
  ta: 'Tamil',
  te: 'Telugu',
};

const SCRIPT_LANGS = new Set(['hi', 'zh', 'ja', 'ko', 'ar', 'ta', 'te']);
const FEW_SHOT = {
  hi: '{"lang":"en","repaired":"What\'s going on?","translation":"क्या हो रहा है?"}',
  en: '{"lang":"hi","repaired":"क्या हो रहा है?","translation":"What\'s going on?"}',
  es: '{"lang":"en","repaired":"What\'s going on?","translation":"¿Qué está pasando?"}',
};

/** True when Whisper already labeled this as the target language. */
function isAlreadyTargetLang(srcLangHint, to) {
  const a = String(srcLangHint || '').toLowerCase();
  const b = String(to || '').toLowerCase();
  if (!a || a === 'und' || !b) return false;
  return a === b || a.startsWith(b) || b.startsWith(a);
}

function lightRepair(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.!?])/g, '$1')
    .trim();
}

/** Heuristic: does `text` look like it is written in target language `to`? */
function looksLikeTargetLang(text, to) {
  const t = String(text || '');
  const code = String(to || '').toLowerCase().slice(0, 2);
  if (!t.trim()) return false;
  if (code === 'hi') return /[\u0900-\u097F]/.test(t);
  if (code === 'zh') return /[\u4e00-\u9fff]/.test(t);
  if (code === 'ja') return /[\u3040-\u30ff\u4e00-\u9fff]/.test(t);
  if (code === 'ko') return /[\uac00-\ud7af]/.test(t);
  if (code === 'ar') return /[\u0600-\u06ff]/.test(t);
  if (code === 'ta') return /[\u0b80-\u0bff]/.test(t);
  if (code === 'te') return /[\u0c00-\u0c7f]/.test(t);
  if (code === 'en') return !/[\u0900-\u097F\u4e00-\u9fff\u0600-\u06ff]/.test(t);
  return !/[\u0900-\u097F\u4e00-\u9fff\u0600-\u06ff\uac00-\ud7af]/.test(t);
}

function tokenBudgetFor(text, to) {
  const n = String(text || '').length;
  const scripted = SCRIPT_LANGS.has(String(to || '').slice(0, 2));
  // Short Discord lines should stay cheap; long lines get more room.
  const floor = scripted ? 96 : 72;
  const mult = scripted ? 1.25 : 1;
  return Math.min(280, Math.max(floor, Math.ceil((n / 3.5) * mult) + 40));
}

/**
 * @param {{text: string, to?: string, srcLangHint?: string, recentContext?: string, disableThinking?: boolean, strict?: boolean}} opts
 */
function buildTranslatePrompt({
  text,
  to = 'en',
  srcLangHint = '',
  recentContext = '',
  disableThinking = false,
  strict = false,
}) {
  const targetName = LANG_NAMES[to] || to;
  const hint = srcLangHint && srcLangHint !== 'und' ? `ASR lang guess: ${srcLangHint}.` : '';
  const shot = FEW_SHOT[to] || FEW_SHOT.en;
  const ctx = recentContext ? `Context: ${recentContext}\n` : '';
  const strictLine = strict
    ? `RETRY: previous answer was NOT in ${targetName}. For hi use Devanagari only. Never English/Indonesian/romanization when to=hi.\n`
    : '';

  // Put /no_think first — Qwen3 follows it more reliably at the top.
  return `${disableThinking ? `${NO_THINK_DIRECTIVE}\n` : ''}Translate meeting ASR into ${targetName} (${to}). Fast, literal.
${TRANSCRIPTION_CAVEAT}
${hint} ${strictLine}${ctx}Example: ${shot}
ASR: """${text}"""
JSON only: {"lang":"<src>","repaired":"<source>","translation":"<${to}>"}`;
}

function parseTranslateJson(raw, to = 'en') {
  const text = String(raw || '').trim();
  if (!text) return null;
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fence ? fence[1].trim() : text;
  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start < 0 || end <= start) return null;
  try {
    const obj = JSON.parse(candidate.slice(start, end + 1));
    if (!obj || typeof obj !== 'object') return null;
    const translationField =
      obj.translation ||
      obj.translated ||
      (String(to).slice(0, 2) === 'en' ? obj.en : '') ||
      '';
    return {
      lang: String(obj.lang || obj.source_lang || 'und').trim().toLowerCase().slice(0, 8) || 'und',
      repaired: String(obj.repaired || obj.clean || obj.source || '').trim(),
      translation: String(translationField).trim(),
    };
  } catch {
    return null;
  }
}

async function runGenerate(llm, prompt, maxTokens) {
  const parts = [];
  let raw = '';
  for await (const token of llm.generate(prompt, {
    maxTokens,
    temperature: TRANSLATE_TEMPERATURE,
  })) {
    parts.push(token);
    raw += token;
    // Same answer, fewer tokens: stop once a complete JSON object is on the wire.
    if (raw.length >= 24 && jsonObjectComplete(raw)) break;
  }
  return parts.length === 1 ? parts[0] : parts.join('');
}

/** True when `s` contains a fully closed top-level `{...}` (string-aware). */
function jsonObjectComplete(s) {
  const start = s.indexOf('{');
  if (start < 0) return false;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return true;
    }
  }
  return false;
}

/**
 * @param llm RunAnywhere LLMModel
 * @param opts {{text: string, to?: string, srcLangHint?: string, recentContext?: string, disableThinking?: boolean}}
 */
async function translateUtterance(llm, opts) {
  const text = String(opts.text || '').trim();
  const to = (opts.to || 'en').toLowerCase();
  if (!text) {
    return {
      lang: opts.srcLangHint || 'und',
      repaired: '',
      translation: '',
      raw: '',
      fast: true,
      targetOk: false,
    };
  }

  if (!ALWAYS_LLM && isAlreadyTargetLang(opts.srcLangHint, to) && looksLikeTargetLang(text, to)) {
    const cleaned = lightRepair(text);
    return {
      lang: opts.srcLangHint || to,
      repaired: cleaned,
      translation: cleaned,
      raw: '',
      fast: true,
      targetOk: true,
    };
  }

  return serialize(async () => {
    const budget = tokenBudgetFor(text, to);
    const baseOpts = {
      text,
      to,
      srcLangHint: opts.srcLangHint || '',
      recentContext: opts.recentContext || '',
      disableThinking: !!opts.disableThinking,
    };

    let raw = await runGenerate(llm, buildTranslatePrompt(baseOpts), budget);
    let parsed = parseTranslateJson(raw, to);

    // Retry only when we clearly got a wrong-script translation (or garbage JSON
    // on a substantial utterance). Do NOT retry empty/noise — that doubles latency.
    const wrongScript =
      parsed &&
      parsed.translation &&
      !looksLikeTargetLang(parsed.translation, to) &&
      !isAlreadyTargetLang(parsed.lang, to);
    const garbageJson = !parsed && text.length >= 24;
    if (wrongScript || garbageJson) {
      const retryRaw = await runGenerate(
        llm,
        buildTranslatePrompt({ ...baseOpts, strict: true, recentContext: '' }),
        Math.min(280, budget + 48)
      );
      const retryParsed = parseTranslateJson(retryRaw, to);
      if (retryParsed && retryParsed.translation) {
        raw = retryRaw;
        parsed = retryParsed;
      }
    }

    if (parsed && (parsed.translation || parsed.repaired)) {
      const translation = parsed.translation || '';
      const targetOk =
        !!translation &&
        (looksLikeTargetLang(translation, to) || isAlreadyTargetLang(parsed.lang, to));
      return {
        lang: parsed.lang || opts.srcLangHint || 'und',
        repaired: parsed.repaired || text,
        translation,
        raw,
        fast: false,
        targetOk,
      };
    }

    return {
      lang: opts.srcLangHint || 'und',
      repaired: text,
      translation: '',
      raw,
      fast: false,
      targetOk: false,
    };
  });
}

module.exports = {
  buildTranslatePrompt,
  parseTranslateJson,
  translateUtterance,
  isAlreadyTargetLang,
  looksLikeTargetLang,
  lightRepair,
  jsonObjectComplete,
  TRANSLATE_MAX_TOKENS,
};
