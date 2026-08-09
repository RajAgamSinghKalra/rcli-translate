// Live repair + translate: one LLM call turns raw multilingual ASR into a
// cleaned source string and a translation into the user's target language.
const { serialize, NO_THINK_DIRECTIVE, TRANSCRIPTION_CAVEAT } = require('./llm');

const TRANSLATE_MAX_TOKENS =
  Number(process.env.RCLI_XL8_TRANSLATE_TOKENS || process.env.RCLI_MEET_TRANSLATE_TOKENS) || 140;
const TRANSLATE_TEMPERATURE = 0.15;
const ALWAYS_LLM = /^(1|on|true|yes)$/i.test(process.env.RCLI_XL8_ALWAYS_LLM || '');

/** True when Whisper already labeled this as the target language. */
function isAlreadyTargetLang(srcLangHint, to) {
  const a = String(srcLangHint || '').toLowerCase();
  const b = String(to || '').toLowerCase();
  if (!a || a === 'und' || !b) return false;
  return a === b || (a.startsWith(b) || b.startsWith(a));
}

/**
 * Lightweight cleanup without a full translate when source ≈ target.
 */
function lightRepair(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.!?])/g, '$1')
    .trim();
}

/**
 * @param {{text: string, to?: string, srcLangHint?: string, recentContext?: string, disableThinking?: boolean}} opts
 */
function buildTranslatePrompt({
  text,
  to = 'en',
  srcLangHint = '',
  recentContext = '',
  disableThinking = false,
}) {
  const hint = srcLangHint
    ? `Whisper guessed the source language code as "${srcLangHint}" (may be wrong — trust the text).`
    : 'Source language is unknown — detect it from the text.';
  const ctx = recentContext
    ? `\nRecent meeting context (for disambiguation only):\n${recentContext}\n`
    : '';

  return `You repair live meeting speech-to-text and translate it into the user's language.

${TRANSCRIPTION_CAVEAT}

${hint}
The speaker may switch languages between utterances — always detect THIS utterance's language, then translate into "${to}".
Target language code: ${to}
${ctx}
Raw ASR utterance:
"""${text}"""

Return ONLY a single JSON object (no markdown fences, no commentary) with keys:
  "lang"         — ISO 639-1 source language code for THIS utterance (e.g. "hi", "en", "es")
  "repaired"     — cleaned version in the SOURCE language (fix ASR errors; keep meaning; do not translate here)
  "translation"  — natural ${to} translation of the repaired meaning

If the utterance is already in ${to}, set lang accordingly, repaired ≈ cleaned original, and translation ≈ repaired.
If the text is empty or pure noise, use lang "und", repaired "", translation "".
${disableThinking ? `\n${NO_THINK_DIRECTIVE}` : ''}`;
}

/** Extract first JSON object from model output (tolerates preamble / fences). */
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

/**
 * @param llm RunAnywhere LLMModel
 * @param opts {{text: string, to?: string, srcLangHint?: string, recentContext?: string, disableThinking?: boolean}}
 * @returns {Promise<{lang: string, repaired: string, translation: string, raw: string}>}
 */
async function translateUtterance(llm, opts) {
  const text = String(opts.text || '').trim();
  const to = (opts.to || 'en').toLowerCase();
  if (!text) {
    return { lang: opts.srcLangHint || 'und', repaired: '', translation: '', raw: '', fast: true };
  }

  // Snappy path: already in target language → skip LLM (override with RCLI_XL8_ALWAYS_LLM=1).
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

  const prompt = buildTranslatePrompt({
    text,
    to,
    srcLangHint: opts.srcLangHint || '',
    recentContext: opts.recentContext || '',
    disableThinking: !!opts.disableThinking,
  });

  return serialize(async () => {
    let raw = '';
    for await (const token of llm.generate(prompt, {
      maxTokens: TRANSLATE_MAX_TOKENS,
      temperature: TRANSLATE_TEMPERATURE,
    })) {
      raw += token;
    }
    const parsed = parseTranslateJson(raw);
    if (parsed && (parsed.translation || parsed.repaired)) {
      return {
        lang: parsed.lang || opts.srcLangHint || 'und',
        repaired: parsed.repaired || text,
        translation: parsed.translation || parsed.repaired || text,
        raw,
        fast: false,
      };
    }
    // Fallback: treat ASR as already-target-language if JSON parse fails.
    return {
      lang: opts.srcLangHint || 'und',
      repaired: text,
      translation: text,
      raw,
      fast: false,
    };
  });
}

module.exports = {
  buildTranslatePrompt,
  parseTranslateJson,
  translateUtterance,
  isAlreadyTargetLang,
  lightRepair,
  TRANSLATE_MAX_TOKENS,
};
