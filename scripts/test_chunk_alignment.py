#!/usr/bin/env python3
"""Test chunk_sec values that align with encoder window boundaries.

encoder window = 128,000 samples = 8.0s
Test values that evenly divide 8.0s vs those that don't.
"""

import ctypes
import json
import struct
import wave
from pathlib import Path

DYLIB = Path.home() / "Github/typeless/Frameworks/qwen-asr/lib/libqwen_asr.dylib"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
FIXTURES = Path.home() / "Github/typeless/Tests/fixtures"

TARGET_IDS = {"zh_long_01", "long_30s_01", "long_60s_01"}
STREAM_CHUNK = 32000  # 2s push chunk


def normalize(text):
    return "".join(text.lower().split())

def levenshtein(a, b):
    if len(a) < len(b): return levenshtein(b, a)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]

def cer(expected, actual):
    ref, hyp = normalize(expected), normalize(actual)
    if not ref: return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)

def load_wav(path):
    with wave.open(path, "rb") as wf:
        nc, sw, fr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nf)
    if sw == 2:
        n = len(raw) // 2
        samples = [s / 32768.0 for s in struct.unpack(f"<{n}h", raw)]
    elif sw == 4:
        n = len(raw) // 4
        samples = list(struct.unpack(f"<{n}f", raw))
    else:
        raise ValueError(f"sw={sw}")
    if nc == 2:
        samples = [(samples[i] + samples[i+1]) / 2.0 for i in range(0, len(samples), 2)]
    if fr != 16000:
        ratio = 16000 / fr
        new_len = int(len(samples) * ratio)
        resampled = []
        for i in range(new_len):
            src = i / ratio; idx = int(src)
            if idx >= len(samples) - 1: resampled.append(samples[-1])
            else: frac = src - idx; resampled.append(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        samples = resampled
    return samples

def load_entries():
    entries = []
    for jp in (FIXTURES / "corpus.json", FIXTURES / "real_manifest.json"):
        data = json.load(open(jp))
        items = data.get("entries", data)
        for e in items:
            if e.get("id") not in TARGET_IDS: continue
            af = e.get("audio_files", {})
            rel = af.get("edge_tts") or af.get("real") or af.get("synthetic")
            if rel: entries.append({"id": e["id"], "wav": str(FIXTURES / rel), "expected": e.get("expected_text", "")})
    return sorted(entries, key=lambda e: e["id"])

def setup_lib():
    lib = ctypes.cdll.LoadLibrary(str(DYLIB))
    lib.qwen_asr_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
    lib.qwen_asr_load_model.restype = ctypes.c_void_p
    lib.qwen_asr_free.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free.restype = None
    lib.qwen_asr_free_string.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_free_string.restype = None
    lib.qwen_asr_set_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qwen_asr_set_language.restype = ctypes.c_int32
    lib.qwen_asr_stream_new.argtypes = []
    lib.qwen_asr_stream_new.restype = ctypes.c_void_p
    lib.qwen_asr_stream_free.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_stream_free.restype = None
    lib.qwen_asr_stream_push.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
    lib.qwen_asr_stream_push.restype = ctypes.c_void_p
    lib.qwen_asr_stream_get_result.argtypes = [ctypes.c_void_p]
    lib.qwen_asr_stream_get_result.restype = ctypes.c_void_p
    lib.qwen_asr_stream_set_chunk_sec.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.qwen_asr_stream_set_chunk_sec.restype = None
    lib.qwen_asr_stream_set_rollback.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.qwen_asr_stream_set_rollback.restype = None
    lib.qwen_asr_stream_set_unfixed_chunks.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.qwen_asr_stream_set_unfixed_chunks.restype = None
    lib.qwen_asr_stream_set_max_new_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.qwen_asr_stream_set_max_new_tokens.restype = None
    return lib

def get_str(lib, ptr):
    if not ptr: return ""
    s = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")
    lib.qwen_asr_free_string(ptr)
    return s

def stream_run(lib, engine, samples):
    stream = lib.qwen_asr_stream_new()
    off = 0
    while off < len(samples):
        end = min(off + STREAM_CHUNK, len(samples))
        chunk = samples[off:end]
        arr = (ctypes.c_float * len(chunk))(*chunk)
        ptr = lib.qwen_asr_stream_push(engine, stream, arr, len(chunk), 0)
        if ptr: lib.qwen_asr_free_string(ptr)
        off = end
    silence = (ctypes.c_float * 1600)()
    ptr = lib.qwen_asr_stream_push(engine, stream, silence, 1600, 1)
    if ptr: lib.qwen_asr_free_string(ptr)
    ptr = lib.qwen_asr_stream_get_result(stream)
    result = get_str(lib, ptr)
    lib.qwen_asr_stream_free(stream)
    return result

def main():
    entries = load_entries()
    print(f"Entries: {[e['id'] for e in entries]}")
    print(f"dylib: {DYLIB}\n")
    lib = setup_lib()

    audio_cache = {}
    for e in entries:
        audio_cache[e["id"]] = load_wav(e["wav"])
        print(f"  {e['id']}: {len(audio_cache[e['id']])/16000:.1f}s")
    print()

    # encoder window = 128,000 samples = 8.0s
    # Test chunk_sec values: aligned vs unaligned
    configs = [
        # Baseline
        ("A. 默认 chunk=2.0 (8.0/2.0=4.0 整除)", 2.0),

        # Aligned values (evenly divide 8.0s)
        ("B. chunk=1.0 (8.0/1.0=8.0 整除)", 1.0),
        ("C. chunk=0.5 (8.0/0.5=16.0 整除)", 0.5),
        ("D. chunk=1.6 (8.0/1.6=5.0 整除)", 1.6),
        ("E. chunk=4.0 (8.0/4.0=2.0 整除)", 4.0),

        # Unaligned values (don't evenly divide 8.0s)
        ("F. chunk=1.5 (8.0/1.5=5.33 不整除)", 1.5),
        ("G. chunk=1.7 (8.0/1.7=4.71 不整除)", 1.7),
        ("H. chunk=1.9 (8.0/1.9=4.21 不整除)", 1.9),
        ("I. chunk=1.8 (8.0/1.8=4.44 不整除)", 1.8),
    ]

    print(f"{'='*80}")
    print(f"encoder window = 128,000 samples = 8.0s")
    print(f"测试假设: 整除 encoder window 的 chunk_sec 不会退化")
    print(f"{'='*80}\n")

    results = {}
    for name, cs in configs:
        print(f"{'='*70}")
        print(f"{name}")
        print(f"  chunk_samples = {cs} × 16000 = {int(cs * 16000)}")
        print(f"  128000 / {int(cs * 16000)} = {128000 / (cs * 16000):.2f}")
        print(f"{'='*70}")

        engine = lib.qwen_asr_load_model(str(MODEL_DIR).encode(), 0, 0)
        lib.qwen_asr_set_language(engine, b"chinese")
        lib.qwen_asr_stream_set_chunk_sec(engine, ctypes.c_float(cs))
        lib.qwen_asr_stream_set_rollback(engine, 5)
        lib.qwen_asr_stream_set_unfixed_chunks(engine, 2)
        lib.qwen_asr_stream_set_max_new_tokens(engine, 32)

        results[name] = {}
        for e in entries:
            samples = audio_cache[e["id"]]
            result = stream_run(lib, engine, samples)
            c = cer(e["expected"], result)
            results[name][e["id"]] = (c, result)
            tag = "OK" if c <= 0.10 else ("WARN" if c <= 0.30 else "HIGH")
            print(f"  [{tag:4s}] {e['id']:15s} CER={c:.3f}  ({len(result)}字)")

        lib.qwen_asr_free(engine)
        print()

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY: chunk_sec 对齐 vs 非对齐")
    print(f"{'='*90}")
    eids = [e["id"] for e in entries]
    header = f"{'Config':<55s}" + "".join(f" {eid:>12s}" for eid in eids) + "  AVG"
    print(header)
    print("-" * len(header))
    for name in [c[0] for c in configs]:
        cols = f"{name:<55s}"
        cers = []
        for eid in eids:
            c = results[name][eid][0]
            cers.append(c)
            marker = " *" if c > 0.10 else "  "
            cols += f" {c:>10.3f}{marker}"
        avg = sum(cers) / len(cers)
        marker = " *" if avg > 0.10 else "  "
        cols += f" {avg:>5.3f}{marker}"
        print(cols)

    print()
    print("(* = CER > 0.10)")
    print()
    print("如果 '整除' 组全部 OK，'不整除' 组全部退化 → 对齐是唯一因素")
    print("如果 '整除' 的小 chunk_sec 也退化 → chunk 大小本身也是因素")


if __name__ == "__main__":
    main()
