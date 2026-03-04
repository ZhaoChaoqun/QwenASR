# MLX Python vs Rust 流式实现对比报告

*调查时间：2026-03-03*

## 背景

`~/Github/mlx-qwen-asr-streaming` 是基于 MLX 框架的 Python 流式 ASR 实现，对长音频没有截断问题。
`~/Github/QwenASR` 的 Rust 实现（`stream_push_audio`）在长音频上出现严重截断（CER=0.871）。

本报告对比两个实现的关键差异，定位截断根因。

---

## 1. 架构对比

| 维度 | MLX Python | Rust (`stream_push_audio`) |
|------|-----------|---------------------------|
| 文件 | `engine.py`, `stabilizer.py`, `state.py` | `transcribe.rs` L941-1360 |
| 推理框架 | MLX (Apple Silicon GPU) | 自研 CPU kernels |
| KV Cache | 每个 chunk 重建 fresh cache | LCP 复用 (prefill embedding 级别对比) |
| 编码器缓存 | 8s window 缓存 + 每 chunk 重编码 partial | 相同策略 |
| Token 稳定化 | byte-level alignment（处理 BPE 重编码） | token-level only |

---

## 2. 关键差异

### 差异 1 [Critical] — MLX 没有 `speech_ended` 检测，Rust 有

**这是截断的直接原因。**

**MLX Python (`engine.py:104-214`)**:

```python
while True:
    remaining = len(state.audio_buffer) - state.audio_cursor
    if remaining < self.chunk_samples and not finalize:
        break
    if remaining == 0:
        break

    # ... encode, decode, stabilize ...

    state.chunk_idx += 1

    if is_final:
        break
```

循环退出条件**只有三个**：
1. 剩余音频不足一个 chunk 且非 finalize
2. 剩余音频为 0
3. `is_final = True`（finalize 且所有音频已处理）

**decoder 产生 EOT 不会导致循环退出**。`_decode()` 返回空 `chunk_tokens` 时，代码继续执行 degeneracy 检测、re-anchor、token 稳定化，然后进入下一个 chunk。

**Rust (`transcribe.rs:1190-1356`)**:

```rust
let speech_ended = chunk_tokens.is_empty()
    && !state.raw_tokens.is_empty()
    && state.chunk_idx >= unfixed_chunks;

// ... emit rollback tokens ...

if speech_ended {
    break;  // ← 丢弃后续所有音频！
}
```

当 decoder 产生 EOT（`chunk_tokens` 为空），**整个 while 循环直接 break**，后续未处理的音频全部丢弃。

**影响**：长音频在 re-anchor 后 decoder 遇到 silence 间隙时产生 EOT → break → 后续几十秒音频丢失。

---

### 差异 2 [Critical] — Re-anchor 时 MLX 截断 audio_buffer，Rust 不截断

**MLX Python (`stabilizer.py:231-234`)**:

```python
# Trim audio buffer so the encoder won't immediately exceed the
# reanchor threshold again on the next chunk.
if keep_audio_samples > 0 and state.audio_cursor > keep_audio_samples:
    trim_start = state.audio_cursor - keep_audio_samples
    state.audio_buffer = state.audio_buffer[trim_start:]
    state.audio_cursor = keep_audio_samples
```

Re-anchor 时**裁剪 audio buffer**，只保留最近 `chunk_samples * 4` 个 samples（~8s）。这确保下次 encoder 不会立即再次超过 `REANCHOR_ENC_SEQ_THRESHOLD`，避免持续触发 re-anchor。

同时 `enc_cache_audio_end` 也随之对齐：
```python
state.enc_cache_audio_end = min(state.enc_cache_audio_end, state.audio_cursor)
```

**Rust (`transcribe.rs:1276-1308`)**:

```rust
if should_reanchor {
    // ... carry tokens, clear prefill cache, reset degen state ...
    // Trim encoder cache to 1 window
    let keep_windows = 1.min(state.enc_cache.len());
    let drop_windows = state.enc_cache.len() - keep_windows;
    if drop_windows > 0 {
        // ... drop old encoder windows ...
    }
    // NOTE: audio_cursor 和 audio_buf/samples 不做任何裁剪！
}
```

Rust 的 re-anchor **不裁剪 audio buffer**（`state.audio_cursor` 不变，`samples` 是外部传入的完整 buffer）。虽然 encoder cache 被清理，但下一个 chunk 执行时，encoder 需要从 `enc_cache_base_windows` 开始重新编码已处理过的 audio window，encoder 序列长度很快再次超过阈值，导致频繁触发 re-anchor。

**影响**：Rust 可能在长音频后半段陷入"re-anchor → encoder 序列立即再超标 → 再次 re-anchor"的循环，大量 CPU 时间浪费在重复编码上，且每次 re-anchor 都可能因 speech_ended 而 break。

---

### 差异 3 [Important] — `max_new_tokens` 差异巨大

| 参数 | MLX Python | Rust |
|------|:----------:|:----:|
| `max_new_tokens` | **128** | **32** |

MLX 允许每个 chunk 最多生成 128 个 token，Rust 限制为 32。

**影响**：对于信息密度高的音频段（如快速语音），32 tokens 可能不足以覆盖 2s chunk 中的所有语音内容。当 decoder 碰到 `max_new_tokens` 上限时，chunk_tokens 被截断，与真正的 speech_ended（decoder 主动产生 EOT）不同，但 Rust 的 degeneracy 和 rollback 逻辑会受到影响。

---

### 差异 4 [Important] — `REANCHOR_ENC_SEQ_THRESHOLD` 差异

| 参数 | MLX Python | Rust |
|------|:----------:|:----:|
| `REANCHOR_ENC_SEQ_THRESHOLD` | **400** (~30s) | **200** (~15s) |
| `REANCHOR_INTERVAL_CHUNKS` | **90** (~180s) | **45** (~90s) |

MLX 的 re-anchor 阈值是 Rust 的 2 倍。这意味着：
- MLX 在 encoder 累积 ~30s 音频的输出后才触发 re-anchor
- Rust 在 ~15s 就触发

更频繁的 re-anchor 意味着更多的 decoder 状态重置，更多机会触发 speech_ended break。

---

### 差异 5 [Moderate] — KV Cache 策略

**MLX**: 每个 chunk 重建 fresh KV cache（`engine.py:328`）。

```python
cache = self.model.make_cache()  # Fresh cache each chunk
```

**Rust**: 通过 LCP (Longest Common Prefix) 复用之前的 KV cache。

```rust
// Compare current and previous prefill embeddings row by row
while reused_prefill < cmp_len {
    if a != b { break; }
    reused_prefill += 1;
}
ctx.kv_cache.len = reused_prefill;  // Reuse this many positions
```

LCP 机制在 re-anchor 后（`prev_prefill_embeds` 被清空）会 fall back 到全量 prefill，与 MLX 行为一致。正常 chunk 间 LCP 可以大幅减少计算量，但对截断问题没有直接影响。

---

### 差异 6 [Moderate] — Byte-level alignment（BPE 重编码修复）

**MLX** (`stabilizer.py:67-101`): 在 compute_stable_tokens 中检测 token 序列 mismatch，用 byte-level 累加找到正确的 emit 位置。

```python
if mismatch:
    stable_bytes = byte_decoder.tokens_to_bytes(list(stable_text_tokens))
    stable_byte_len = len(stable_bytes)
    cum_bytes = 0
    for j in range(len(candidate_tokens)):
        cum_bytes += len(byte_decoder._token_to_bytes(candidate_tokens[j]))
        if cum_bytes >= stable_byte_len:
            aligned_emit_from = j + 1
            break
    emit_from = max(emit_from, aligned_emit_from)
```

**Rust**: 没有这个机制。Re-anchor 后 BPE 重新编码 carry tokens 可能产生不同的 token 序列，导致 `stable_text_tokens` 与 `candidate_tokens` 的对齐偏移，可能在 UTF-8 字符中间截断，产生乱码碎片。

这解释了 Swift benchmark 日志中出现的碎片拼接现象（如"容器云计算平台"被截断为不连贯的碎片）。

---

### 差异 7 [Minor] — Audio buffer trimming on re-anchor

| 行为 | MLX | Rust |
|------|-----|------|
| audio buffer 裁剪 | 保留最近 `chunk_samples * 4` | 不裁剪（外部 buffer 不可变） |
| enc_cache_audio_end 重置 | 对齐到裁剪后的 cursor | 不重置 |
| enc_cache_base_windows 调整 | 通过 cache 清理隐含处理 | 仅调整 base_windows offset |

MLX 的 audio buffer 裁剪是一个关键设计：它确保 re-anchor 后 encoder 从一个较短的 audio 位置开始，不会立即再次超过阈值。Rust 由于 `stream_push_audio` 接收的是外部传入的完整 buffer（`&[f32]`），无法修改它，只能通过 `enc_cache_base_windows` 跳过已编码的 window。但如果 `enc_cache_base_windows` 的跳转不够，encoder 仍然需要重新编码大量 audio。

---

## 3. 差异汇总表

| # | 差异 | MLX Python | Rust | 严重性 | 与截断的关系 |
|---|------|-----------|------|:------:|-------------|
| 1 | speech_ended 处理 | **无**——decoder EOT 不终止循环 | **break** 整个 while 循环 | 🔴 Critical | 直接原因 |
| 2 | re-anchor audio 裁剪 | 裁剪 buffer 到最近 ~8s | 不裁剪 | 🔴 Critical | 导致频繁 re-anchor |
| 3 | max_new_tokens | 128 | 32 | 🟡 Important | 限制每 chunk 输出容量 |
| 4 | REANCHOR 阈值 | 400 tokens (~30s) | 200 tokens (~15s) | 🟡 Important | Rust 更频繁触发 re-anchor |
| 5 | KV cache 策略 | 每 chunk fresh | LCP 复用 | 🟢 Moderate | 性能差异，不直接影响截断 |
| 6 | byte-level alignment | 有 | 无 | 🟢 Moderate | 影响 re-anchor 后的乱码碎片 |
| 7 | reanchor interval | 90 chunks (~180s) | 45 chunks (~90s) | 🟡 Important | Rust 更频繁触发 re-anchor |

---

## 4. 修复建议（按优先级）

### P0: 移除 `speech_ended` break

参考 MLX 实现：decoder 产生 EOT 时，不应终止整个处理循环。应该让 while 循环继续处理后续 chunk。

如果需要保留 "speech ended" 的语义（例如实时麦克风场景），应该：
- 仅在外部调用者传入 `finalize=true` 且音频处理完毕时才 break
- 对于非 finalize 的 speech_ended，执行与 re-anchor 相同的状态重置（carry tokens + clear cache）

### P1: Re-anchor 时裁剪 audio context

参考 MLX 的 `perform_reanchor(state, keep_audio_samples=chunk_samples * 4)`：

由于 `stream_push_audio` 的 audio buffer 是外部传入的引用（`&[f32]`），不能直接裁剪。但可以通过调整 `state.audio_cursor` 来实现等效效果：

- 记录 re-anchor 时的 `audio_cursor` 位置
- 让后续 encoder 只处理 re-anchor 点之后的 audio
- 这需要调整 `enc_cache_base_windows` 和全 end 计算逻辑

或者，在 `c_api.rs` 的 `QwenAsrStreamState` 中维护一个 `audio_trim_offset`，让 encoder 跳过已裁剪的部分。

### P2: 增大 `REANCHOR_ENC_SEQ_THRESHOLD`

从 200 增大到 400（与 MLX 一致），减少 re-anchor 频率。同时将 `REANCHOR_INTERVAL_CHUNKS` 从 45 增大到 90。

### P3: 增大 `max_new_tokens`

从 32 增大到 128（与 MLX 一致），允许 decoder 在信息密度高的段落生成更多 token。

### P4: 实现 byte-level alignment

参考 MLX 的 `compute_stable_tokens()` 中的 byte-level alignment 逻辑，在 Rust 的 token stabilization 中加入 BPE 重编码检测和 byte 对齐，避免 re-anchor 后的乱码碎片。

---

## 5. 关键代码位置参考

### MLX Python

| 功能 | 文件 | 行号 |
|------|------|------|
| 主循环（无 speech_ended） | `engine.py` | L104-214 |
| Re-anchor（含 audio trim） | `stabilizer.py` | L195-251 |
| Degeneracy 检测 | `stabilizer.py` | L114-137 |
| Token 稳定化（byte alignment） | `stabilizer.py` | L21-111 |
| Decoder（fresh cache） | `engine.py` | L318-361 |
| Encoder 缓存 | `engine.py` | L272-316 |

### Rust

| 功能 | 文件 | 行号 |
|------|------|------|
| 主循环（含 speech_ended break） | `transcribe.rs` | L982-1357 |
| speech_ended 检测 + break | `transcribe.rs` | L1190-1356 |
| Re-anchor（无 audio trim） | `transcribe.rs` | L1270-1308 |
| Degeneracy 检测 | `transcribe.rs` | L1226-1268 |
| Token 稳定化（无 byte alignment） | `transcribe.rs` | L1311-1347 |
| Decoder（LCP cache 复用） | `transcribe.rs` | L1129-1181 |
| Encoder 缓存 | `transcribe.rs` | L992-1059 |
