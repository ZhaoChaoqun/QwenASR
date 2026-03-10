#!/usr/bin/env python3
"""Compare streaming CER and performance with different unfixed_chunks values.

Usage:
    uv run scripts/compare_unfixed_chunks.py
"""

import ctypes
import json
import os
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DYLIB = REPO_ROOT / "target" / "release" / "libqwen_asr.dylib"
MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots"
    / "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
)

FIXTURES = Path.home() / "Github" / "typeless" / "Tests" / "fixtures"
CORPUS_JSON = FIXTURES / "corpus.json"
REAL_MANIFEST_JSON = FIXTURES / "real_manifest.json"

SKIP_CATEGORIES = {"silence", "silence_short", "hallucination"}
STREAM_CHUNK_SAMPLES = 32000  # 2s @ 16kHz


# ── CER ──────────────────────────────────────────────────────────────────────

def cer(expected: str, actual: str) -> float:
    e = expected.lower().replace(" ", "")
    a = actual.lower().replace(" ", "")
    if not e:
        return 0.0 if not a else 1.0
    n, m = len(e), len(a)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if e[i - 1] == a[j - 1] else 1
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
    return dp[m] / n


# ── Audio loading ────────────────────────────────────────────────────────────

def load_wav_samples(wav_path: str) -> list[float]:
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())
    import struct
    shorts = struct.unpack(f"<{len(frames)//2}h", frames)
    return [s / 32768.0 for s in shorts]


def load_entries() -> list[dict]:
    entries = []
    for json_path in (CORPUS_JSON, REAL_MANIFEST_JSON):
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        for e in data["entries"]:
            if e.get("category") in SKIP_CATEGORIES:
                continue
            audio_files = e.get("audio_files", {})
            rel = (
                audio_files.get("edge_tts")
                or audio_files.get("real")
                or audio_files.get("synthetic")
            )
            if not rel:
                continue
            wav = str(FIXTURES / rel)
            if not os.path.isfile(wav):
                continue
            entries.append({
                "id": e["id"],
                "category": e.get("category", "unknown"),
                "expected_text": e.get("expected_text", ""),
                "wav": wav,
                "duration_sec": e.get("duration_sec", 0),
            })
    return entries


# ── Rust engine ──────────────────────────────────────────────────────────────

class RustEngine:
    def __init__(self, model_path: str, n_threads: int = 0):
        lib = ctypes.CDLL(str(DYLIB))

        lib.qwen_asr_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
        lib.qwen_asr_load_model.restype = ctypes.c_void_p
        lib.qwen_asr_free.argtypes = [ctypes.c_void_p]
        lib.qwen_asr_free.restype = None
        lib.qwen_asr_free_string.argtypes = [ctypes.c_void_p]
        lib.qwen_asr_free_string.restype = None
        lib.qwen_asr_set_use_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.qwen_asr_set_use_gpu.restype = None
        lib.qwen_asr_transcribe_pcm.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32
        ]
        lib.qwen_asr_transcribe_pcm.restype = ctypes.c_void_p
        lib.qwen_asr_set_segment_sec.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.qwen_asr_set_segment_sec.restype = None

        # Streaming API
        lib.qwen_asr_stream_new.argtypes = []
        lib.qwen_asr_stream_new.restype = ctypes.c_void_p
        lib.qwen_asr_stream_free.argtypes = [ctypes.c_void_p]
        lib.qwen_asr_stream_free.restype = None
        lib.qwen_asr_stream_reset.argtypes = [ctypes.c_void_p]
        lib.qwen_asr_stream_reset.restype = None
        lib.qwen_asr_stream_push.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
        ]
        lib.qwen_asr_stream_push.restype = ctypes.c_void_p
        lib.qwen_asr_stream_get_result.argtypes = [ctypes.c_void_p]
        lib.qwen_asr_stream_get_result.restype = ctypes.c_void_p
        lib.qwen_asr_stream_set_unfixed_chunks.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.qwen_asr_stream_set_unfixed_chunks.restype = None

        self._lib = lib
        self._engine = lib.qwen_asr_load_model(model_path.encode("utf-8"), n_threads, 0)
        if not self._engine:
            raise RuntimeError(f"Failed to load model from {model_path}")

    def close(self):
        if self._engine:
            self._lib.qwen_asr_free(self._engine)
            self._engine = None

    def set_use_gpu(self, use_gpu: bool):
        self._lib.qwen_asr_set_use_gpu(self._engine, 1 if use_gpu else 0)

    def set_unfixed_chunks(self, chunks: int):
        self._lib.qwen_asr_stream_set_unfixed_chunks(self._engine, chunks)

    def _get_string(self, ptr) -> str:
        if not ptr:
            return ""
        text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.qwen_asr_free_string(ptr)
        return text

    def transcribe_stream(self, samples: list[float]) -> str:
        stream = self._lib.qwen_asr_stream_new()
        if not stream:
            raise RuntimeError("Failed to create stream state")
        try:
            offset = 0
            while offset < len(samples):
                end = min(offset + STREAM_CHUNK_SAMPLES, len(samples))
                chunk = samples[offset:end]
                arr = (ctypes.c_float * len(chunk))(*chunk)
                ptr = self._lib.qwen_asr_stream_push(self._engine, stream, arr, len(chunk), 0)
                if ptr:
                    self._lib.qwen_asr_free_string(ptr)
                offset = end
            silence = (ctypes.c_float * 1600)()
            ptr = self._lib.qwen_asr_stream_push(self._engine, stream, silence, 1600, 1)
            if ptr:
                self._lib.qwen_asr_free_string(ptr)
            ptr = self._lib.qwen_asr_stream_get_result(stream)
            return self._get_string(ptr)
        finally:
            self._lib.qwen_asr_stream_free(stream)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_streaming(engine: RustEngine, entries: list[dict], label: str):
    """Run streaming benchmark and return results dict."""
    results = {}
    n = len(entries)
    for i, e in enumerate(entries, 1):
        eid = e["id"]
        print(f"  [{i:3d}/{n}] {eid:<30s} ", end="", flush=True)
        try:
            samples = load_wav_samples(e["wav"])
            t0 = time.monotonic()
            text = engine.transcribe_stream(samples)
            elapsed = time.monotonic() - t0
            c = cer(e["expected_text"], text)
            results[eid] = {"text": text, "elapsed": elapsed, "cer": c}
            print(f"CER={c:.3f}  ({elapsed:.1f}s)")
        except Exception as ex:
            print(f"ERROR: {ex}")
            results[eid] = {"text": "", "elapsed": 0.0, "cer": 1.0}
    return results


def main():
    entries = load_entries()
    total_audio_sec = sum(e["duration_sec"] for e in entries)
    print(f"Loaded {len(entries)} entries, total audio: {total_audio_sec:.1f}s\n")

    engine = RustEngine(str(MODEL_DIR))
    engine.set_use_gpu(True)

    # Warmup
    print("Warmup...")
    engine.transcribe_stream([0.0] * 16000)
    print("Warmup done.\n")

    configs = [
        ("unfixed_chunks=2 (default)", 2),
        ("unfixed_chunks=0", 0),
    ]

    all_results = {}
    for label, uc in configs:
        print(f"{'=' * 60}")
        print(f"Running: {label}")
        print(f"{'=' * 60}")
        engine.set_unfixed_chunks(uc)
        all_results[label] = run_streaming(engine, entries, label)
        print()

    engine.close()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("Summary: unfixed_chunks comparison (GPU streaming)")
    print(f"{'=' * 70}")
    print(f"{'Pipeline':<35s} {'Avg CER':>8s} {'CER=0':>7s} {'CER≤0.10':>9s} {'CER>0.20':>9s} {'Time':>7s} {'RTF':>7s}")
    print("-" * 70)
    for label, _ in configs:
        res = all_results[label]
        cers = [res[e["id"]]["cer"] for e in entries]
        avg_cer = sum(cers) / len(cers)
        cer_0 = sum(1 for c in cers if c == 0.0)
        cer_10 = sum(1 for c in cers if c <= 0.10)
        cer_hi = sum(1 for c in cers if c > 0.20)
        total_time = sum(res[e["id"]]["elapsed"] for e in entries)
        rtf = total_time / total_audio_sec
        print(f"{label:<35s} {avg_cer:>8.4f} {cer_0:>5d}/67 {cer_10:>7d} {cer_hi:>9d} {total_time:>6.1f}s {rtf:>6.3f}x")

    # ── Per-entry comparison (show differences) ──
    print(f"\n{'=' * 70}")
    print("Per-entry CER differences (unfixed=0 vs unfixed=2)")
    print(f"{'=' * 70}")
    label_default = configs[0][0]
    label_zero = configs[1][0]
    diffs = []
    for e in entries:
        eid = e["id"]
        c_default = all_results[label_default][eid]["cer"]
        c_zero = all_results[label_zero][eid]["cer"]
        if abs(c_default - c_zero) > 0.001:
            diffs.append((eid, c_default, c_zero, c_zero - c_default))

    if not diffs:
        print("No CER differences found between the two configurations.")
    else:
        diffs.sort(key=lambda x: x[3])
        print(f"{'ID':<30s} {'uc=2':>8s} {'uc=0':>8s} {'Δ':>8s}  Note")
        print("-" * 70)
        for eid, c2, c0, delta in diffs:
            note = "better" if delta < 0 else "worse"
            print(f"{eid:<30s} {c2:>8.3f} {c0:>8.3f} {delta:>+8.3f}  {note}")
        print(f"\nTotal entries with difference: {len(diffs)}")
        better = sum(1 for _, _, _, d in diffs if d < 0)
        worse = sum(1 for _, _, _, d in diffs if d > 0)
        print(f"  Better with uc=0: {better}")
        print(f"  Worse with uc=0:  {worse}")


if __name__ == "__main__":
    main()
