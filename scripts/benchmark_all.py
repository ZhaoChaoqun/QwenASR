#!/usr/bin/env python3
"""Unified benchmark: Rust (GPU/CPU, offline/stream) + MLX (stream).

Runs up to 5 modes on the same 67-audio test set and generates a side-by-side
Markdown report.

Usage:
    uv run scripts/benchmark_all.py
    uv run scripts/benchmark_all.py --modes rust_gpu_offline,mlx_stream
"""

import argparse
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
MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots"
    / "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
)

MLX_PROJECT = Path.home() / "Github" / "mlx-qwen-asr-streaming"

FIXTURES = Path.home() / "Github" / "typeless" / "Tests" / "fixtures"
CORPUS_JSON = FIXTURES / "corpus.json"
REAL_MANIFEST_JSON = FIXTURES / "real_manifest.json"

OUTPUT_REPORT = REPO_ROOT / "docs" / "benchmark-report-all.md"

SKIP_CATEGORIES = {"silence", "silence_short", "hallucination"}
STREAM_CHUNK_SAMPLES = 32000  # 2s @ 16kHz

ALL_MODES = [
    "rust_gpu_offline",
    "rust_gpu_stream",
    "rust_cpu_offline",
    "rust_cpu_stream",
    "mlx_stream",
]

MODE_LABELS = {
    "rust_gpu_offline": "Rust GPU 离线",
    "rust_gpu_stream": "Rust GPU 流式",
    "rust_cpu_offline": "Rust CPU 离线",
    "rust_cpu_stream": "Rust CPU 流式",
    "mlx_stream": "MLX 流式",
}


# ── CER ──────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return "".join(text.lower().split())


def levenshtein(a: str, b: str) -> int:
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
    ref = normalize(expected)
    hyp = normalize(actual)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


# ── WAV loader ───────────────────────────────────────────────────────────────

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


# ── Corpus loader ────────────────────────────────────────────────────────────

def load_entries() -> list[dict]:
    entries = []
    for json_path in (CORPUS_JSON, REAL_MANIFEST_JSON):
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
                print(f"WARNING: audio not found: {wav}", file=sys.stderr)
                continue
            entries.append(
                {
                    "id": e["id"],
                    "category": e.get("category", "unknown"),
                    "expected_text": e.get("expected_text", ""),
                    "wav": wav,
                    "duration_sec": e.get("duration_sec", 0),
                }
            )
    return entries


# ── Rust C API engine ────────────────────────────────────────────────────────

class RustEngine:
    def __init__(self, model_dir: str, n_threads: int = 0):
        lib = ctypes.cdll.LoadLibrary(str(DYLIB))

        lib.qwen_asr_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
        lib.qwen_asr_load_model.restype = ctypes.c_void_p

        lib.qwen_asr_free.argtypes = [ctypes.c_void_p]
        lib.qwen_asr_free.restype = None

        lib.qwen_asr_free_string.argtypes = [ctypes.c_void_p]
        lib.qwen_asr_free_string.restype = None

        lib.qwen_asr_transcribe_pcm.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ]
        lib.qwen_asr_transcribe_pcm.restype = ctypes.c_void_p

        lib.qwen_asr_set_segment_sec.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.qwen_asr_set_segment_sec.restype = None

        lib.qwen_asr_set_use_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.qwen_asr_set_use_gpu.restype = None

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

        self._lib = lib
        self._engine = lib.qwen_asr_load_model(model_dir.encode("utf-8"), n_threads, 0)
        if not self._engine:
            raise RuntimeError(f"Failed to load model from {model_dir}")

    def close(self):
        if self._engine:
            self._lib.qwen_asr_free(self._engine)
            self._engine = None

    def set_use_gpu(self, use_gpu: bool):
        self._lib.qwen_asr_set_use_gpu(self._engine, 1 if use_gpu else 0)

    def _get_string(self, ptr) -> str:
        if not ptr:
            return ""
        text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        self._lib.qwen_asr_free_string(ptr)
        return text

    def transcribe_offline(self, samples: list[float]) -> str:
        arr = (ctypes.c_float * len(samples))(*samples)
        self._lib.qwen_asr_set_segment_sec(self._engine, ctypes.c_float(20.0))
        ptr = self._lib.qwen_asr_transcribe_pcm(self._engine, arr, len(samples))
        self._lib.qwen_asr_set_segment_sec(self._engine, ctypes.c_float(0.0))
        return self._get_string(ptr)

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


# ── MLX engine (runs via subprocess in MLX project's venv) ───────────────────

MLX_PYTHON = MLX_PROJECT / ".venv" / "bin" / "python"
MLX_BENCHMARK = MLX_PROJECT / "scripts" / "benchmark.py"


def run_mlx_benchmark(entries: list[dict], model_path: str) -> dict:
    """Run MLX benchmark via subprocess using MLX project's venv.

    Returns {id: {"text": str, "elapsed": float, "cer": float}}.
    """
    import subprocess
    import tempfile

    output_json = Path(tempfile.mktemp(suffix=".json"))

    cmd = [
        str(MLX_PYTHON), str(MLX_BENCHMARK),
        "--model", model_path,
        "--output", str(output_json),
    ]
    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=False, text=True, cwd=str(MLX_PROJECT))
    if proc.returncode != 0:
        print(f"ERROR: MLX benchmark exited with code {proc.returncode}", file=sys.stderr)
        # Return dummy results
        return {e["id"]: {"text": "", "elapsed": 0.0, "cer": 1.0} for e in entries}

    with open(output_json) as f:
        mlx_data = json.load(f)
    output_json.unlink(missing_ok=True)

    # Convert MLX results format to our format
    results = {}
    for r in mlx_data.get("results", []):
        results[r["id"]] = {
            "text": r.get("result", ""),
            "elapsed": r.get("elapsed", 0.0),
            "cer": r.get("cer", 1.0),
        }

    # Fill in any missing entries
    for e in entries:
        if e["id"] not in results:
            results[e["id"]] = {"text": "", "elapsed": 0.0, "cer": 1.0}

    return results


# ── Report generator ─────────────────────────────────────────────────────────

def truncate(text: str, maxlen: int = 60) -> str:
    if len(text) <= maxlen:
        return text
    return text[:maxlen] + "..."


def generate_report(entries: list[dict], results: dict, modes: list[str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(entries)
    total_audio_sec = sum(e["duration_sec"] for e in entries)
    mode_labels = [MODE_LABELS[m] for m in modes]

    lines = []
    lines.append("# ASR Pipeline 综合对比评估报告")
    lines.append("")
    lines.append(f"*生成时间：{now}*")
    lines.append(f"*测试集：{n} 条音频（corpus.json + real_manifest.json）*")
    lines.append(f"*Pipeline：{', '.join(mode_labels)}*")
    lines.append("")
    lines.append("**CER 计算方式**：保留标点符号，仅做 lower + 去空格后计算字符错误率。")
    lines.append("")
    lines.append(f"**测试环境**：Apple Silicon (aarch64), 总音频时长 {total_audio_sec:.1f}s")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 1: Overall ──
    lines.append("## 1. 总体 CER + RTF 汇总")
    lines.append("")
    header = "| Pipeline | 平均 CER | CER=0 条数 | CER≤0.10 | CER≤0.20 | CER>0.20 | 总推理时长 | RTF |"
    sep = "|----------|:-------:|:---------:|:-------:|:-------:|:-------:|:---------:|:---:|"
    lines.append(header)
    lines.append(sep)

    for mode in modes:
        label = MODE_LABELS[mode]
        cers = [results[mode][e["id"]]["cer"] for e in entries]
        avg_cer = sum(cers) / len(cers) if cers else 0
        cer_0 = sum(1 for c in cers if c == 0.0)
        cer_10 = sum(1 for c in cers if c <= 0.10)
        cer_20 = sum(1 for c in cers if c <= 0.20)
        cer_hi = sum(1 for c in cers if c > 0.20)
        total_time = sum(results[mode][e["id"]]["elapsed"] for e in entries)
        rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0
        lines.append(
            f"| {label} | {avg_cer:.4f} | {cer_0}/{n} | {cer_10} | {cer_20} | {cer_hi} | {total_time:.1f}s | {rtf:.3f}x |"
        )

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

    mode_cols = " | ".join(MODE_LABELS[m] for m in modes)
    header2 = f"| 类别 | 条数 | {mode_cols} |"
    sep2 = "|------|:----:|" + "|".join(":------:" for _ in modes) + "|"
    lines.append(header2)
    lines.append(sep2)
    for cat in cat_order:
        cat_entries = cats[cat]
        nc = len(cat_entries)
        avgs = []
        for mode in modes:
            avg = sum(results[mode][e["id"]]["cer"] for e in cat_entries) / nc
            avgs.append(f"{avg:.3f}")
        lines.append(f"| {cat} | {nc} | {' | '.join(avgs)} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 3: Per-entry ──
    lines.append("## 3. 逐条 CER 详细")
    lines.append("")
    mode_cols_h = " | ".join(MODE_LABELS[m] for m in modes)
    lines.append(f"| # | ID | {mode_cols_h} | 期望文本 |")
    sep3 = "|---|-----|" + "|".join(":------:" for _ in modes) + "|------|"
    lines.append(sep3)
    for idx, e in enumerate(entries, 1):
        eid = e["id"]
        vals = " | ".join(f"{results[m][eid]['cer']:.3f}" for m in modes)
        exp = truncate(e["expected_text"])
        lines.append(f"| {idx} | {eid} | {vals} | {exp} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 4: High CER ──
    lines.append("## 4. 高 CER 条目详情 (CER > 0.20)")
    lines.append("")
    for mode in modes:
        label = MODE_LABELS[mode]
        high = [
            (e, results[mode][e["id"]]["cer"])
            for e in entries
            if results[mode][e["id"]]["cer"] > 0.20
        ]
        high.sort(key=lambda x: -x[1])
        lines.append(f"### {label}")
        lines.append("")
        if not high:
            lines.append("*无 CER > 0.20 的条目。*")
        else:
            lines.append("| # | ID | CER | 期望文本 | 实际输出 |")
            lines.append("|---|-----|:---:|---------|---------|")
            for i, (e, c) in enumerate(high, 1):
                eid = e["id"]
                exp = truncate(e["expected_text"], 80)
                act = truncate(results[mode][eid]["text"], 80)
                lines.append(f"| {i} | {eid} | {c:.3f} | {exp} | {act} |")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def run_mode(mode: str, entries: list[dict], rust_engine: "RustEngine | None"):
    """Run one benchmark mode and return {id: {text, elapsed, cer}}."""
    results = {}
    n = len(entries)

    is_stream = "stream" in mode
    use_gpu = "gpu" in mode

    rust_engine.set_use_gpu(use_gpu)

    for i, e in enumerate(entries, 1):
        eid = e["id"]
        print(f"  [{i:3d}/{n}] {eid:<30s} ", end="", flush=True)
        try:
            samples = load_wav_samples(e["wav"])
            t0 = time.monotonic()
            if is_stream:
                text = rust_engine.transcribe_stream(samples)
            else:
                text = rust_engine.transcribe_offline(samples)
            elapsed = time.monotonic() - t0
            c = cer(e["expected_text"], text)
            results[eid] = {"text": text, "elapsed": elapsed, "cer": c}
            print(f"CER={c:.3f}  ({elapsed:.1f}s)")
        except Exception as ex:
            print(f"ERROR: {ex}")
            results[eid] = {"text": "", "elapsed": 0.0, "cer": 1.0}

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified ASR benchmark (Rust + MLX)")
    parser.add_argument(
        "--modes",
        default=",".join(ALL_MODES),
        help=f"Comma-separated modes to run. Available: {','.join(ALL_MODES)}",
    )
    parser.add_argument("--mlx-model", default="mlx-community/Qwen3-ASR-0.6B-8bit")
    parser.add_argument("--output", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    for m in modes:
        if m not in ALL_MODES:
            print(f"ERROR: unknown mode '{m}'. Available: {ALL_MODES}", file=sys.stderr)
            sys.exit(1)

    rust_modes = [m for m in modes if not m.startswith("mlx")]
    mlx_modes = [m for m in modes if m.startswith("mlx")]

    # Load test corpus
    print("Loading test corpus...")
    entries = load_entries()
    print(f"Loaded {len(entries)} entries.")

    # Initialize engines
    rust_engine = None

    if rust_modes:
        if not DYLIB.is_file():
            print(f"ERROR: dylib not found at {DYLIB}", file=sys.stderr)
            print("Run `cargo build --release -p qwen-asr --features macos-ffi,metal --lib` first.", file=sys.stderr)
            sys.exit(1)
        if not MODEL_DIR.is_dir():
            print(f"ERROR: model dir not found at {MODEL_DIR}", file=sys.stderr)
            sys.exit(1)
        print("Loading Rust model via C API...")
        rust_engine = RustEngine(str(MODEL_DIR))
        print("Rust model loaded.")
        # Warmup: run a short transcription to initialize GPU shaders
        print("Warmup (GPU shader compilation)...")
        warmup_samples = [0.0] * 16000  # 1s silence
        rust_engine.set_use_gpu(True)
        rust_engine.transcribe_offline(warmup_samples)
        rust_engine.set_use_gpu(False)
        rust_engine.transcribe_offline(warmup_samples)
        print("Warmup done.")

    if mlx_modes:
        if not MLX_PROJECT.is_dir():
            print(f"ERROR: MLX project not found at {MLX_PROJECT}", file=sys.stderr)
            sys.exit(1)
        if not MLX_PYTHON.is_file():
            print(f"ERROR: MLX venv not found at {MLX_PYTHON}", file=sys.stderr)
            sys.exit(1)

    # Run benchmarks
    all_results = {}
    for mode in modes:
        label = MODE_LABELS[mode]
        print(f"\n{'='*60}")
        print(f"Running {label} ({len(entries)} files)...")
        print(f"{'='*60}")
        if mode.startswith("mlx"):
            all_results[mode] = run_mlx_benchmark(entries, args.mlx_model)
        else:
            all_results[mode] = run_mode(mode, entries, rust_engine)

    # Cleanup
    if rust_engine:
        rust_engine.close()

    # Generate report
    print("\nGenerating report...")
    report = generate_report(entries, all_results, modes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report written to {args.output}")

    # Print summary
    total_audio_sec = sum(e["duration_sec"] for e in entries)
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for mode in modes:
        label = MODE_LABELS[mode]
        cers = [all_results[mode][e["id"]]["cer"] for e in entries]
        avg_cer = sum(cers) / len(cers) if cers else 0
        total_time = sum(all_results[mode][e["id"]]["elapsed"] for e in entries)
        rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0
        print(f"  {label:<20s}  CER={avg_cer:.4f}  RTF={rtf:.3f}x  ({total_time:.1f}s)")


if __name__ == "__main__":
    main()
