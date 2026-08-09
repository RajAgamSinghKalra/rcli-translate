// Cosine-similarity search over every transcript segment for the whole
// session (not just the recency window) -- lets a question pull in a moment
// from long before the last-N-minutes window, e.g. "combine what she said at
// the start with what he said just now".
function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * @param embedder RunAnywhere embedder (embed() returns an L2-normalized Float32Array)
 * @param onError {(message: string) => void} non-fatal embed failures
 */
function createRetrieval(embedder, { onError = () => {} } = {}) {
  const items = [];
  const pending = []; // segments waiting to be embedded (deferred during live translate)
  let flushTimer = null;
  let flushing = false;

  function embedOne(segment) {
    try {
      const vec = embedder.embed(segment.text);
      items.push({ ...segment, vec });
      return true;
    } catch (err) {
      onError(`could not index a transcript segment for search: ${err.message}`);
      return false;
    }
  }

  function scheduleFlush() {
    if (flushTimer || flushing) return;
    flushTimer = setTimeout(() => {
      flushTimer = null;
      void flushPending();
    }, 0);
    if (flushTimer.unref) flushTimer.unref();
  }

  async function flushPending() {
    if (flushing) return;
    flushing = true;
    try {
      // Yield between embeds so Whisper/LLM keep the event loop.
      while (pending.length) {
        const seg = pending.shift();
        embedOne(seg);
        await new Promise((r) => setImmediate(r));
      }
    } finally {
      flushing = false;
      if (pending.length) scheduleFlush();
    }
  }

  return {
    /**
     * @param segment {{line: string, text: string, elapsedMs: number}}
     * @param opts {{defer?: boolean}} defer=true queues embed off the live path
     * @returns {boolean} whether the segment was accepted (indexed or queued)
     */
    add(segment, opts = {}) {
      if (opts.defer) {
        pending.push(segment);
        scheduleFlush();
        return true;
      }
      return embedOne(segment);
    },

    /** Force any deferred embeds (call on stop / before Q&A). */
    flush() {
      if (flushTimer) {
        clearTimeout(flushTimer);
        flushTimer = null;
      }
      while (pending.length) embedOne(pending.shift());
      return pending.length === 0;
    },

    get size() {
      return items.length + pending.length;
    },

    get pendingCount() {
      return pending.length;
    },

    /**
     * Top-k segments by cosine similarity to `query` (embeddings are already
     * L2-normalized, so a dot product IS the cosine similarity).
     * @param exclude {Set<string>} segment lines to omit (e.g. already shown
     *   verbatim in the recency window) so the prompt doesn't repeat itself.
     */
    topK(query, k = 5, exclude = new Set()) {
      // Catch up deferred embeds before searching so Q&A quality stays intact.
      this.flush();
      if (items.length === 0) return [];
      let qvec;
      try {
        qvec = embedder.embed(query);
      } catch (err) {
        onError(`could not embed the question for search: ${err.message}`);
        return [];
      }
      return items
        .filter((item) => !exclude.has(item.line))
        .map((item) => ({ item, score: dot(qvec, item.vec) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k)
        .map((s) => s.item);
    },
  };
}

module.exports = { createRetrieval, dot };
