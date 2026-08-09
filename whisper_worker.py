"""Persistent Vulkan Whisper worker for rcli-translate.

Loads whisper.dll + ggml-vulkan.dll once, keeps large-v3-turbo warm on the GPU,
and transcribes float32 PCM utterances from stdin.

Protocol v3 (binary, little-endian):
  request:
    uint32 n_samples
    if n_samples == 0: shutdown
    else:
      uint8  mode        # 0 = partial (greedy, fast), 1 = final (beam, accurate)
      uint32 prompt_len
      bytes  prompt      # utf-8 conditioning text (may be empty)
      uint32 lang_len
      bytes  lang        # e.g. b"en" or b"auto" (empty → CLI --language)
      float32[n_samples] # 16 kHz mono PCM
  response:
    uint32 n_bytes | utf-8 JSON {"lang":"xx","text":"..."}
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import sys
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_void_p,
)


SAMPLE_RATE = 16000
WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1


class WhisperAheads(ctypes.Structure):
    _fields_ = [
        ("n_heads", c_size_t),
        ("heads", c_void_p),
    ]


class WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("flash_attn", c_bool),
        ("gpu_device", c_int),
        ("dtw_token_timestamps", c_bool),
        ("dtw_aheads_preset", c_int),
        ("dtw_n_top", c_int),
        ("dtw_aheads", WhisperAheads),
        ("dtw_mem_size", c_size_t),
    ]


class WhisperVadParams(ctypes.Structure):
    _fields_ = [
        ("threshold", c_float),
        ("min_speech_duration_ms", c_int),
        ("min_silence_duration_ms", c_int),
        ("max_speech_duration_s", c_float),
        ("speech_pad_ms", c_int),
        ("samples_overlap", c_float),
    ]


class Greedy(ctypes.Structure):
    _fields_ = [("best_of", c_int)]


class BeamSearch(ctypes.Structure):
    _fields_ = [("beam_size", c_int), ("patience", c_float)]


class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", c_int),
        ("n_threads", c_int),
        ("n_max_text_ctx", c_int),
        ("offset_ms", c_int),
        ("duration_ms", c_int),
        ("translate", c_bool),
        ("no_context", c_bool),
        ("no_timestamps", c_bool),
        ("single_segment", c_bool),
        ("print_special", c_bool),
        ("print_progress", c_bool),
        ("print_realtime", c_bool),
        ("print_timestamps", c_bool),
        ("token_timestamps", c_bool),
        ("thold_pt", c_float),
        ("thold_ptsum", c_float),
        ("max_len", c_int),
        ("split_on_word", c_bool),
        ("max_tokens", c_int),
        ("debug_mode", c_bool),
        ("audio_ctx", c_int),
        ("tdrz_enable", c_bool),
        ("suppress_regex", c_char_p),
        ("initial_prompt", c_char_p),
        ("carry_initial_prompt", c_bool),
        ("prompt_tokens", c_void_p),
        ("prompt_n_tokens", c_int),
        ("language", c_char_p),
        ("detect_language", c_bool),
        ("suppress_blank", c_bool),
        ("suppress_nst", c_bool),
        ("temperature", c_float),
        ("max_initial_ts", c_float),
        ("length_penalty", c_float),
        ("temperature_inc", c_float),
        ("entropy_thold", c_float),
        ("logprob_thold", c_float),
        ("no_speech_thold", c_float),
        ("greedy", Greedy),
        ("beam_search", BeamSearch),
        ("new_segment_callback", c_void_p),
        ("new_segment_callback_user_data", c_void_p),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("encoder_begin_callback", c_void_p),
        ("encoder_begin_callback_user_data", c_void_p),
        ("abort_callback", c_void_p),
        ("abort_callback_user_data", c_void_p),
        ("logits_filter_callback", c_void_p),
        ("logits_filter_callback_user_data", c_void_p),
        ("grammar_rules", c_void_p),
        ("n_grammar_rules", c_size_t),
        ("i_start_rule", c_size_t),
        ("grammar_penalty", c_float),
        ("vad", c_bool),
        ("vad_model_path", c_char_p),
        ("vad_params", WhisperVadParams),
    ]


def _load_lib(bin_dir: str):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_dir)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return ctypes.CDLL(os.path.join(bin_dir, "whisper.dll"))


def _setup_api(lib):
    lib.whisper_context_default_params_by_ref.restype = POINTER(WhisperContextParams)
    lib.whisper_free_context_params.argtypes = [POINTER(WhisperContextParams)]

    lib.whisper_init_from_file_with_params.restype = c_void_p
    lib.whisper_init_from_file_with_params.argtypes = [c_char_p, WhisperContextParams]

    lib.whisper_full_default_params_by_ref.restype = POINTER(WhisperFullParams)
    lib.whisper_full_default_params_by_ref.argtypes = [c_int]
    lib.whisper_free_params.argtypes = [POINTER(WhisperFullParams)]

    lib.whisper_full.restype = c_int
    lib.whisper_full.argtypes = [c_void_p, WhisperFullParams, POINTER(c_float), c_int]

    lib.whisper_full_n_segments.restype = c_int
    lib.whisper_full_n_segments.argtypes = [c_void_p]

    lib.whisper_full_get_segment_text.restype = c_char_p
    lib.whisper_full_get_segment_text.argtypes = [c_void_p, c_int]

    lib.whisper_full_lang_id.restype = c_int
    lib.whisper_full_lang_id.argtypes = [c_void_p]

    lib.whisper_lang_str.restype = c_char_p
    lib.whisper_lang_str.argtypes = [c_int]

    lib.whisper_free.argtypes = [c_void_p]
    lib.whisper_print_system_info.restype = c_char_p


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed")
        buf.extend(chunk)
    return bytes(buf)


def _write_text(stream, text: str) -> None:
    data = text.encode("utf-8")
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def _transcribe(lib, ctx, samples, n_samples, lang_bytes, prompt_bytes, mode, n_threads):
    strategy = WHISPER_SAMPLING_BEAM_SEARCH if mode == 1 else WHISPER_SAMPLING_GREEDY
    params_ptr = lib.whisper_full_default_params_by_ref(strategy)
    params = params_ptr.contents
    params.n_threads = n_threads
    params.translate = False
    params.no_context = True
    params.no_timestamps = True
    # Partials: force one segment for snappy growing text.
    # Finals: allow natural segments for longer utterances.
    params.single_segment = mode == 0
    params.print_special = False
    params.print_progress = False
    params.print_realtime = False
    params.print_timestamps = False
    lang_str = (lang_bytes or b"en").decode("utf-8", errors="replace").strip().lower() or "en"
    auto = lang_str in ("auto", "detect", "")
    params.language = b"auto" if auto else lang_str.encode("utf-8")
    params.detect_language = auto
    params.initial_prompt = prompt_bytes
    params.carry_initial_prompt = True
    params.temperature = 0.0
    params.temperature_inc = 0.0  # don't climb -- better for accented speech
    params.suppress_blank = True
    params.suppress_nst = True
    # Slightly stricter on partials (less hallucinated filler mid-stream).
    params.no_speech_thold = 0.7 if mode == 0 else 0.55
    if mode == 1:
        params.beam_search.beam_size = 5
        params.beam_search.patience = 1.0
    else:
        params.greedy.best_of = 1

    rc = lib.whisper_full(ctx, params, samples, n_samples)
    lib.whisper_free_params(params_ptr)
    if rc != 0:
        return {"lang": "und", "text": ""}

    detected = lang_str if not auto else "und"
    try:
        lang_id = lib.whisper_full_lang_id(ctx)
        if lang_id >= 0:
            name = lib.whisper_lang_str(lang_id)
            if name:
                detected = name.decode("utf-8", errors="replace")
    except Exception:
        pass

    parts = []
    n_seg = lib.whisper_full_n_segments(ctx)
    for i in range(n_seg):
        t = lib.whisper_full_get_segment_text(ctx, i)
        if t:
            parts.append(t.decode("utf-8", errors="replace"))
    return {"lang": detected or "und", "text": "".join(parts).strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument(
        "--prompt",
        default="Live meeting speech. Transcribe accurately in the spoken language.",
    )
    parser.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()

    lib = _load_lib(args.bin_dir)
    _setup_api(lib)

    info = lib.whisper_print_system_info()
    if info:
        sys.stderr.write(f"[whisper-worker] {info.decode('utf-8', errors='replace')}\n")
        sys.stderr.flush()

    cparams_ptr = lib.whisper_context_default_params_by_ref()
    if not cparams_ptr:
        sys.stderr.write("[whisper-worker] failed to get context params\n")
        return 1
    cparams = cparams_ptr.contents
    cparams.use_gpu = True
    cparams.flash_attn = True
    cparams.gpu_device = 0

    sys.stderr.write(f"[whisper-worker] loading model on GPU: {args.model}\n")
    sys.stderr.flush()
    ctx = lib.whisper_init_from_file_with_params(args.model.encode("utf-8"), cparams)
    lib.whisper_free_context_params(cparams_ptr)
    if not ctx:
        sys.stderr.write("[whisper-worker] whisper_init_from_file_with_params failed\n")
        return 1
    sys.stderr.write("[whisper-worker] ready\n")
    sys.stderr.flush()

    default_lang = args.language.encode("utf-8")
    default_prompt = args.prompt.encode("utf-8")
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        while True:
            header = _read_exact(stdin, 4)
            (n_samples,) = struct.unpack("<I", header)
            if n_samples == 0:
                break

            meta = _read_exact(stdin, 5)
            mode = meta[0]
            (prompt_len,) = struct.unpack("<I", meta[1:5])
            prompt_bytes = _read_exact(stdin, prompt_len) if prompt_len else default_prompt
            if not prompt_bytes:
                prompt_bytes = default_prompt

            lang_len_raw = _read_exact(stdin, 4)
            (lang_len,) = struct.unpack("<I", lang_len_raw)
            lang_bytes = _read_exact(stdin, lang_len) if lang_len else default_lang
            if not lang_bytes:
                lang_bytes = default_lang

            raw = _read_exact(stdin, n_samples * 4)
            samples = (c_float * n_samples).from_buffer_copy(raw)

            result = _transcribe(
                lib, ctx, samples, n_samples, lang_bytes, prompt_bytes, mode, args.threads
            )
            # Compact JSON — Node parses lang + text from every response.
            _write_text(stdout, json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    except EOFError:
        pass
    finally:
        lib.whisper_free(ctx)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
