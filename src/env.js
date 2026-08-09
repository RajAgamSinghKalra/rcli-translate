// Env helpers: prefer RCLI_XL8_*, fall back to RCLI_MEET_* so a shared
// RunAnywhere / Whisper setup from rcli-meet still works.
function env(name, fallback = '') {
  const xl8 = process.env[`RCLI_XL8_${name}`];
  if (xl8 !== undefined && xl8 !== '') return xl8;
  const meet = process.env[`RCLI_MEET_${name}`];
  if (meet !== undefined && meet !== '') return meet;
  return fallback;
}

module.exports = { env };
