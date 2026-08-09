// Live repair + translate: one LLM call turns raw multilingual ASR into a
// cleaned source string and a translation into the user's target language.
const { serialize, NO_THINK_DIRECTIVE, TRANSCRIPTION_CAVEAT } = require('./llm');

const TRANSLATE_MAX_TOKENS =
  Number(process.env.RCLI_XL8_TRANSLATE_TOKENS || process.env.RCLI_MEET_TRANSLATE_TOKENS) || 96;
const TRANSLATE_TEMPERATURE = 0.1;
const ALWAYS_LLM = /^(1|on|true|yes)$/i.test(process.env.RCLI_XL8_ALWAYS_LLM || '');

const LANG_NAMES = {
  en: 'English',
  hi: 'Hindi (हिन्दी, Devanagari script)',
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
  // Latin targets: reject if mostly Devanagari/CJK (wrong direction).
  if (code === 'en') return !/[\u0900-\u097F\u4e00-\u9fff]/.test(t);
  return true;
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
  const hint = srcLangHint
    ? `ASR language guess: "${srcLangHint}" (may be wrong — trust the words).`
    : 'Detect the source language from the text.';
  const ctx = recentContext
    ? `\nRecent context:\n${recentContext}\n`
    : '';
  const strictLine = strict
    ? `\nCRITICAL RETRY: Your previous answer was NOT in ${targetName}. translation MUST use the correct script for "${to}". For Hindi use Devanagari only (example: "क्या हो रहा है?"). Never reply in English/Indonesian/romanization when to=hi.\n`
    : '';

  return `Translate live meeting ASR into ${targetName}. Be fast and literal.

${TRANSCRIPTION_CAVEAT}
${hint}
Source may change per utterance. Target code: ${to} = ${targetName}
${strictLine}${ctx}
ASR:
"""${text}"""

Return ONLY JSON:
{"lang":"<source-iso>","repaired":"<cleaned SOURCE text>","translation":"<${to} text>"}
If already ${to}, translation ≈ repaired. If noise: lang "und", empty strings.
${disableThinking ? `\n${NO_THINK_DIRECTIVE}` : ''}`;
}

function parseTranslateJson(raw) {
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
    return {
      lang: String(obj.lang || obj.source_lang || 'und').trim().toLowerCase().slice(0, 8) || 'und',
      repaired: String(obj.repaired || obj.clean || obj.source || '').trim(),
      translation: String(obj.translation || obj.translated || obj.en || '').trim(),
    };
  } catch {
    return null;
  }
}

async function runTranslateOnce(llm, prompt, maxTokens) {
  let raw = '';
  for await (const token of llm.generate(prompt, {
    maxTokens,
    temperature: TRANSLATE_TEMPERATURE,
  })) {
    raw += token;
  }
  return { raw, parsed: parseTranslateJson(raw) };
}

/**
 * @param llm RunAnywhere LLMModel
 * @param opts {{text: string, to?: string, srcLangHint?: string, recentContext?: string, disableThinking?: boolean}}
 */
async function translateUtterance(llm, opts) {
  const text = String(opts.text || '').trim();
  const to = (opts.to || 'en').toLowerCase();
  if (!text) {
    return { lang: opts.srcLangHint || 'und', repaired: '', translation: '', raw: '', fast: true };
  }

  if (!ALWAYS_LLM && isAlreadyTargetLang(opts.srcLangHint, to)) {
    const cleaned = lightRepair(text);
    return {
      lang: opts.srcLangHint || to,
      repaired: cleaned,
      translation: cleaned,
      raw: '',
      fast: true,
    };
  }

  return serialize(async () => {
    const baseOpts = {
      text,
      to,
      srcLangHint: opts.srcLangHint || '',
      recentContext: opts.recentContext || '',
      disableThinking: !!opts.disableThinking,
    };
    let { raw, parsed } = await runTranslateOnce(
      llm,
      buildTranslatePrompt(baseOpts),
      TRANSLATE_MAX_TOKENS
    );

    // Qwen sometimes ignores --to hi and returns English/Indonesian. Retry once hard.
    if (
      parsed &&
      parsed.translation &&
      !isAlreadyTargetLang(parsed.lang, to) &&
      !looksLikeTargetLang(parsed.translation, to)
    ) {
      const retry = await runTranslateOnce(
        llm,
        buildTranslatePrompt({ ...baseOpts, strict: true, recentContext: '' }),
        TRANSLATE_MAX_TOKENS
      );
      if (retry.parsed && retry.parsed.translation && looksLikeTargetLang(retry.parsed.translation, to)) {
        raw = retry.raw;
        parsed = retry.parsed;
      }
    }

    if (parsed && (parsed.translation || parsed.repaired)) {
      return {
        lang: parsed.lang || opts.srcLangHint || 'und',
        repaired: parsed.repaired || text,
        translation: parsed.translation || parsed.repaired || text,
        raw,
        fast: false,
        targetOk: looksLikeTargetLang(parsed.translation || '', to) || isAlreadyTargetLang(parsed.lang, to),
      };
    }
    return {
      lang: opts.srcLangHint || 'und',
      repaired: text,
      translation: text,
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
  TRANSLATE_MAX_TOKENS,
};
