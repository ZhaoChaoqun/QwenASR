#!/usr/bin/env python3
"""Debug streaming ASR with verbose token-level logging.

Runs a single audio file through the streaming C API with verbosity=3
to capture detailed token-level debug information.

Usage:
    uv run scripts/debug_stream.py <wav_file> [verbosity]
    uv run scripts/debug_stream.py ~/Github/typeless/Tests/fixtures/audio/edge_tts/long_60s_01.wav 3
"""

import ctypes
import struct
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DYLIB = REPO_ROOT / "target" / "release" / "libqwen_asr.dylib"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"

STREAM_CHUNK_SAMPLES = 32000  # 2s @ 16kHz


def load_wav_samples(wav_path: str) -> list[float]:
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        count = len(raw) // 2
        samples = list(struct.unpack(f"<{count}h", raw))
        samples = [s / 32768.0 for s in samples]
    elif sample_width == 4:
        count = len(raw) // 4
        samples = list(struct.unpack(f"<{count}f", raw))
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels == 2:
        mono = []
        for i in range(0, len(samples), 2):
            mono.append((samples[i] + samples[i + 1]) / 2.0)
        samples = mono

    if frame_rate != 16000:
        ratio = 16000 / frame_rate
        new_len = int(len(samples) * ratio)
        resampled = []
        for i in range(new_len):
            src = i / ratio
            idx = int(src)
            if idx >= len(samples) - 1:
                resampled.append(samples[-1])
            else:
                frac = src - idx
                resampled.append(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        samples = resampled

    return samples


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <wav_file> [verbosity]", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]
    verbosity = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if not DYLIB.is_file():
        print(f"ERROR: dylib not found: {DYLIB}", file=sys.stderr)
        print("Run: cargo build --release -p qwen-asr --features macos-ffi", file=sys.stderr)
        sys.exit(1)

    lib = ctypes.cdll.LoadLibrary(str(DYLIB))

    lib.qwen_asr_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
    lib.qwen_asr_load_model.restype = ctypes.c_void_p
    lib.qwen_asr_free.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free.restype = None
    lib.qwen_asr_free_string.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free_string.restype = None
    lib.qwen_asr_stream_new.argtypes = []
    lib.qwen_asr_stream_new.restype = ctypes.c_void_p
    lib.qwen_asr_stream_free.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_stream_free.restype = None
    lib.qwen_asr_stream_push.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
    ]
    lib.qwen_asr_stream_push.restype = ctypes.c_void_p
    lib.qwen_asr_stream_get_result.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_stream_get_result.restype = ctypes.c_void_p

    print(f"Loading model (verbosity={verbosity})...", file=sys.stderr)
    engine = lib.qwen_asr_load_model(str(MODEL_DIR).encode(), 0, verbosity)
    if not engine:
        print("ERROR: failed to load model", file=sys.stderr)
        sys.exit(1)

    print(f"Loading audio: {wav_path}", file=sys.stderr)
    samples = load_wav_samples(wav_path)
    duration = len(samples) / 16000.0
    print(f"Audio duration: {duration:.1f}s ({len(samples)} samples)", file=sys.stderr)

    stream = lib.qwen_asr_stream_new()
    if not stream:
        print("ERROR: failed to create stream", file=sys.stderr)
        sys.exit(1)

    # Push audio in 2s chunks
    offset = 0
    push_idx = 0
    while offset < len(samples):
        end = min(offset + STREAM_CHUNK_SAMPLES, len(samples))
        chunk = samples[offset:end]
        arr = (ctypes.c_float * len(chunk))(*chunk)
        ptr = lib.qwen_asr_stream_push(engine, stream, arr, len(chunk), 0)
        delta = ""
        if ptr:
            delta = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
            lib.qwen_asr_free_string(ptr)
        if delta:
            print(f"[push {push_idx}] delta: \"{delta}\"", file=sys.stderr)
        offset = end
        push_idx += 1

    # Finalize with 0.1s silence
    silence = (ctypes.c_float * 1600)()
    ptr = lib.qwen_asr_stream_push(engine, stream, silence, 1600, 1)
    if ptr:
        delta = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        lib.qwen_asr_free_string(ptr)
        if delta:
            print(f"[finalize] delta: \"{delta}\"", file=sys.stderr)

    # Get full result
    ptr = lib.qwen_asr_stream_get_result(stream)
    result = ""
    if ptr:
        result = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        lib.qwen_asr_free_string(ptr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"FINAL RESULT:", file=sys.stderr)
    print(result, file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    lib.qwen_asr_stream_free(stream)
    lib.qwen_asr_free(engine)


if __name__ == "__main__":
    main()
