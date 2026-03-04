#!/usr/bin/env python3
"""Benchmark QwenASR Rust engine against corpus.json + real_manifest.json.

Runs offline and streaming modes, computes CER, and generates a Markdown report
in the same format as the Swift benchmark report.

Streaming mode uses the C API (libqwen_asr.dylib) via ctypes to call
stream_push_audio() — the same code path used by the Swift FFI wrapper.
This ensures the benchmark reflects real streaming quality.
"""

import ctypes
import json
import os
import struct
import sys
import time
import wave
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DYLIB = REPO_ROOT / "target" / "release" / "libqwen_asr.dylib"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"

FIXTURES = Path.home() / "Github/typeless/Tests/fixtures"
CORPUS_JSON = FIXTURES / "corpus.json"
REAL_MANIFEST_JSON = FIXTURES / "real_manifest.json"
AUDIO_BASE = FIXTURES

OUTPUT_REPORT = REPO_ROOT / "docs" / "benchmark-report-rust.md"

# Categories to skip (no expected text / silence)
SKIP_CATEGORIES = {"silence", "silence_short", "hallucination"}

# Streaming chunk size: 2s @ 16kHz (matches Rust default stream_chunk_sec=2.0)
STREAM_CHUNK_SAMPLES = 32000

# ── CER helpers ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lower-case and strip all whitespace (matching Swift CER logic)."""
    return "".join(text.lower().split())


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def cer(expected: str, actual: str) -> float:
    """Character Error Rate: edit_distance / len(reference)."""
    ref = normalize(expected)
    hyp = normalize(actual)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


# ── WAV loader ───────────────────────────────────────────────────────────────

def load_wav_samples(wav_path: str) -> list[float]:
    """Load WAV file and return f32 PCM samples (16kHz mono).

    Handles common formats: 16-bit int, 32-bit float.
    Resamples to 16kHz if needed.
    """
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Convert to float samples
    if sample_width == 2:
        # 16-bit signed int
        count = len(raw) // 2
        samples = list(struct.unpack(f"<{count}h", raw))
        samples = [s / 32768.0 for s in samples]
    elif sample_width == 4:
        # 32-bit float
        count = len(raw) // 4
        samples = list(struct.unpack(f"<{count}f", raw))
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Mix to mono if stereo
    if n_channels == 2:
        mono = []
        for i in range(0, len(samples), 2):
            mono.append((samples[i] + samples[i + 1]) / 2.0)
        samples = mono

    # Simple resample to 16kHz if needed
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


# ── C API wrapper (for streaming) ────────────────────────────────────────────

_lib = None

def _load_lib():
    """Load libqwen_asr.dylib and set up function signatures."""
    global _lib
    if _lib is not None:
        return _lib

    lib = ctypes.cdll.LoadLibrary(str(DYLIB))

    # qwen_asr_load_model
    lib.qwen_asr_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
    lib.qwen_asr_load_model.restype = ctypes.c_void_p

    # qwen_asr_free
    lib.qwen_asr_free.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free.restype = None

    # qwen_asr_free_string
    lib.qwen_asr_free_string.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free_string.restype = None

    # qwen_asr_transcribe_pcm
    lib.qwen_asr_transcribe_pcm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    lib.qwen_asr_transcribe_pcm.restype = ctypes.c_void_p

    # qwen_asr_set_segment_sec
    lib.qwen_asr_set_segment_sec.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.qwen_asr_set_segment_sec.restype = None

    # qwen_asr_set_language
    lib.qwen_asr_set_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qwen_asr_set_language.restype = ctypes.c_int32

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

    _lib = lib
    return lib


class ASREngine:
    """Wrapper around libqwen_asr C API."""

    def __init__(self, model_dir: str, n_threads: int = 0):
        self._lib = _load_lib()
        self._engine = self._lib.qwen_asr_load_model(
            model_dir.encode("utf-8"), n_threads, 0
        )
        if not self._engine:
            raise RuntimeError(f"Failed to load model from {model_dir}")
        self._stream = None

    def close(self):
        if self._stream:
            self._lib.qwen_asr_stream_free(self._stream)
            self._stream = None
        if self._engine:
            self._lib.qwen_asr_free(self._engine)
            self._engine = None

    def _get_string(self, ptr) -> str:
        """Extract string from C heap pointer and free it."""
        if not ptr:
            return ""
        text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.qwen_asr_free_string(ptr)
        return text

    def transcribe_offline(self, samples: list[float]) -> str:
        """Transcribe using offline (non-streaming) API."""
        arr = (ctypes.c_float * len(samples))(*samples)
        # Use segment_sec=20 for long audio (same as Swift)
        self._lib.qwen_asr_set_segment_sec(self._engine, ctypes.c_float(20.0))
        ptr = self._lib.qwen_asr_transcribe_pcm(self._engine, arr, len(samples))
        self._lib.qwen_asr_set_segment_sec(self._engine, ctypes.c_float(0.0))
        return self._get_string(ptr)

    def transcribe_stream(self, samples: list[float]) -> str:
        """Transcribe using streaming API (same code path as Swift FFI).

        Simulates the Swift benchmark's streaming pattern:
        - Push audio in 2s chunks (32000 samples)
        - Push 0.1s silence + finalize
        - Read accumulated result via stream_get_result()
        """
        stream = self._lib.qwen_asr_stream_new()
        if not stream:
            raise RuntimeError("Failed to create stream state")

        try:
            # Push audio in chunks (matching Swift benchmark: 2s chunks)
            offset = 0
            while offset < len(samples):
                end = min(offset + STREAM_CHUNK_SAMPLES, len(samples))
                chunk = samples[offset:end]
                arr = (ctypes.c_float * len(chunk))(*chunk)
                ptr = self._lib.qwen_asr_stream_push(
                    self._engine, stream, arr, len(chunk), 0
                )
                if ptr:
                    self._lib.qwen_asr_free_string(ptr)
                offset = end

            # Push minimal silence (0.1s) + finalize (matching Swift benchmark)
            silence = (ctypes.c_float * 1600)()  # 0.1s @ 16kHz
            ptr = self._lib.qwen_asr_stream_push(
                self._engine, stream, silence, 1600, 1
            )
            if ptr:
                self._lib.qwen_asr_free_string(ptr)

            # Get full accumulated result
            ptr = self._lib.qwen_asr_stream_get_result(stream)
            return self._get_string(ptr)
        finally:
            self._lib.qwen_asr_stream_free(stream)


# ── Corpus loader ────────────────────────────────────────────────────────────

def load_entries() -> list[dict]:
    """Load and merge entries from corpus.json and real_manifest.json."""
    entries = []
    for json_path in (CORPUS_JSON, REAL_MANIFEST_JSON):
        with open(json_path) as f:
            data = json.load(f)
        for e in data["entries"]:
            if e.get("category") in SKIP_CATEGORIES:
                continue
            # Resolve audio path: prefer edge_tts > real > synthetic
            audio_files = e.get("audio_files", {})
            rel = audio_files.get("edge_tts") or audio_files.get("real") or audio_files.get("synthetic")
            if not rel:
                continue
            wav = str(AUDIO_BASE / rel)
            if not os.path.isfile(wav):
                print(f"WARNING: audio not found: {wav}", file=sys.stderr)
                continue
            entries.append({
                "id": e["id"],
                "category": e.get("category", "unknown"),
                "expected_text": e.get("expected_text", ""),
                "wav": wav,
                "duration_sec": e.get("duration_sec", 0),
            })
    return entries


# ── Report generator ─────────────────────────────────────────────────────────

def truncate(text: str, maxlen: int = 60) -> str:
    if len(text) <= maxlen:
        return text
    return text[:maxlen] + "..."


def generate_report(entries: list[dict], results: dict) -> str:
    """Generate markdown report from results.

    results = {
        "offline": {id: {"text": str, "elapsed": float, "cer": float}},
        "stream":  {id: {"text": str, "elapsed": float, "cer": float}},
    }
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(entries)
    total_audio_sec = sum(e["duration_sec"] for e in entries)

    # ── Section 1: Overall ──
    lines = []
    lines.append("# ASR Pipeline 量化对比评估报告 (Rust)")
    lines.append("")
    lines.append(f"*生成时间：{now}*")
    lines.append(f"*测试集：{n} 条音频（corpus.json + real_manifest.json）*")
    lines.append("*Pipeline：Qwen3-ASR (离线), Qwen3-ASR (流式)*")
    lines.append("*运行方式：Rust C API（libqwen_asr.dylib via ctypes）*")
    lines.append("")
    lines.append("**CER 计算方式**：保留标点符号，仅做 lower + 去空格后计算字符错误率。")
    lines.append("")
    lines.append(f"**测试环境**：Apple Silicon (aarch64), 总音频时长 {total_audio_sec:.1f}s")
    lines.append("")
    lines.append("> 注：离线使用 `qwen_asr_transcribe_pcm()`，流式使用 `qwen_asr_stream_push()` + 2s chunk（与 Swift FFI 同一代码路径）。模型仅加载一次。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 总体 CER 汇总")
    lines.append("")

    rows = []
    for mode, label in [("offline", "Qwen3-ASR (离线)"), ("stream", "Qwen3-ASR (流式)")]:
        cers = [results[mode][e["id"]]["cer"] for e in entries]
        avg_cer = sum(cers) / len(cers) if cers else 0
        cer_0 = sum(1 for c in cers if c == 0.0)
        cer_10 = sum(1 for c in cers if c <= 0.10)
        cer_20 = sum(1 for c in cers if c <= 0.20)
        cer_hi = sum(1 for c in cers if c > 0.20)
        total_time = sum(results[mode][e["id"]]["elapsed"] for e in entries)
        rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0
        rows.append(f"| {label} | {avg_cer:.4f} | {cer_0}/{n} | {cer_10} | {cer_20} | {cer_hi} | {total_time:.1f}s | {rtf:.3f}x |")

    lines.append("| Pipeline | 平均 CER | CER=0 条数 | CER≤0.10 | CER≤0.20 | CER>0.20 | 总推理时长 | RTF |")
    lines.append("|----------|:-------:|:---------:|:-------:|:-------:|:-------:|:---------:|:---:|")
    lines.extend(rows)
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 2: By category ──
    lines.append("## 2. 按类别 CER 汇总")
    lines.append("")

    cats = defaultdict(list)
    for e in entries:
        cats[e["category"]].append(e)
    cat_order = sorted(cats.keys())

    lines.append("| 类别 | 条数 | Qwen3-ASR (离线) | Qwen3-ASR (流式) |")
    lines.append("|------|:----:|:------:|:------:|")
    for cat in cat_order:
        cat_entries = cats[cat]
        nc = len(cat_entries)
        off_avg = sum(results["offline"][e["id"]]["cer"] for e in cat_entries) / nc
        str_avg = sum(results["stream"][e["id"]]["cer"] for e in cat_entries) / nc
        lines.append(f"| {cat} | {nc} | {off_avg:.3f} | {str_avg:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 3: Per-entry ──
    lines.append("## 3. 逐条 CER 详细")
    lines.append("")
    lines.append("| # | ID | Qwen3-ASR (离线) | Qwen3-ASR (流式) | 期望文本 |")
    lines.append("|---|-----|:------:|:------:|------|")
    for idx, e in enumerate(entries, 1):
        eid = e["id"]
        off = results["offline"][eid]["cer"]
        stm = results["stream"][eid]["cer"]
        exp = truncate(e["expected_text"])
        lines.append(f"| {idx} | {eid} | {off:.3f} | {stm:.3f} | {exp} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 4: High CER detail ──
    lines.append("## 4. 高 CER 条目详情 (CER > 0.20)")
    lines.append("")

    for mode, label in [("offline", "Qwen3-ASR (离线)"), ("stream", "Qwen3-ASR (流式)")]:
        high = [(e, results[mode][e["id"]]["cer"]) for e in entries if results[mode][e["id"]]["cer"] > 0.20]
        high.sort(key=lambda x: -x[1])

        lines.append(f"### {label}")
        lines.append("")
        if not high:
            lines.append("*无 CER > 0.20 的条目。*")
        else:
            lines.append("| # | ID | CER | 期望文本 | 实际输出 | 分析 |")
            lines.append("|---|-----|:---:|---------|---------|------|")
            for i, (e, c) in enumerate(high, 1):
                eid = e["id"]
                exp = truncate(e["expected_text"], 80)
                act = truncate(results[mode][eid]["text"], 80)
                lines.append(f"| {i} | {eid} | {c:.3f} | {exp} | {act} | |")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Verify dylib exists
    if not DYLIB.is_file():
        print(f"ERROR: dylib not found at {DYLIB}", file=sys.stderr)
        print("Run `cargo build --release --features ios` first.", file=sys.stderr)
        sys.exit(1)
    if not MODEL_DIR.is_dir():
        print(f"ERROR: model dir not found at {MODEL_DIR}", file=sys.stderr)
        sys.exit(1)

    print("Loading test corpus...")
    entries = load_entries()
    print(f"Loaded {len(entries)} entries.")

    print("Loading model via C API (libqwen_asr.dylib)...")
    engine = ASREngine(str(MODEL_DIR))
    print("Model loaded.")

    results = {"offline": {}, "stream": {}}

    for mode in ("offline", "stream"):
        is_stream = mode == "stream"
        label = "流式" if is_stream else "离线"
        print(f"\n{'='*60}")
        print(f"Running {label} mode ({len(entries)} files)...")
        if is_stream:
            print(f"  (using C API stream_push_audio, chunk={STREAM_CHUNK_SAMPLES} samples)")
        print(f"{'='*60}")

        for i, e in enumerate(entries, 1):
            eid = e["id"]
            print(f"  [{i:3d}/{len(entries)}] {eid:<30s} ", end="", flush=True)
            try:
                samples = load_wav_samples(e["wav"])
                t0 = time.monotonic()
                if is_stream:
                    text = engine.transcribe_stream(samples)
                else:
                    text = engine.transcribe_offline(samples)
                elapsed = time.monotonic() - t0
                c = cer(e["expected_text"], text)
                results[mode][eid] = {"text": text, "elapsed": elapsed, "cer": c}
                print(f"CER={c:.3f}  ({elapsed:.1f}s)")
            except Exception as ex:
                print(f"ERROR: {ex}")
                results[mode][eid] = {"text": "", "elapsed": 0.0, "cer": 1.0}

    engine.close()

    # Generate report
    print("\nGenerating report...")
    report = generate_report(entries, results)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Report written to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
