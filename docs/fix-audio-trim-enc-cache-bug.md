# Bug 报告：audio trim 后 enc_cache_base_windows 未更新导致长音频流式转录截断

## 1. 问题现象

使用 Rust C API 流式路径（`stream_push_audio`）转录 60 秒音频时，输出在约 15 秒处截断，后续 45 秒的音频几乎没有产出任何文本。

| 测试用例 | 离线 CER | 流式 CER（修复前） | 流式 CER（修复后） | MLX 流式 CER |
|----------|:--------:|:------------------:|:------------------:|:------------:|
| long_60s_01 (75s) | 0.003 | **0.707** | **0.023** | 0.015 |
| long_30s_01 (43s) | 0.010 | **0.536** | **0.015** | 0.015 |

修复前 long_60s_01 的输出仅为：

> 软件工程是一门研究用工程化方法构建和维护有效的、实用的和高质量的软件的学科。它涉及到程序设计语言、数据库、软件开发工具、系统平台等方面的知识。现代软件开发通常采用敏捷开发方法，强调快速迭代和持续集成和持续

修复后的输出覆盖了完整内容（约 320 字），与离线结果基本一致。

---

## 2. 架构背景

### 2.1 Encoder Window Cache 机制

流式转录中，音频被分为 2 秒的 chunk 推入。encoder 按 **attention window**（每个 window 约 8 秒 = 128000 samples）对音频编码，结果缓存在 `enc_cache` 中复用，避免重复编码。

关键数据结构：

```
audio buffer (samples):
[window 0: 0~128000] [window 1: 128000~256000] [window 2: 256000~384000] [window 3: 384000~512000] ...

enc_cache: [EncWindow { seq_len, enc_output }, ...]   // 缓存的 encoder 输出
enc_cache_base_windows: usize    // 已经从 cache 中 drop 掉的 window 数量
```

encoder 通过以下公式定位下一个要编码的 window 在 audio buffer 中的偏移：

```rust
let ws = (enc_cache_base_windows + enc_cache.len()) * enc_window_samples;
// → samples[ws .. ws + enc_window_samples]
```

`enc_cache_base_windows` 的作用是：当 re-anchor/degen reset 时 drop 掉旧 window 后，记住被 drop 的数量，使后续 window 偏移计算仍然正确。

### 2.2 Re-anchor 与 Audio Trim 机制

当 encoder 总 token 数超过阈值（200 tokens ≈ 15 秒），触发 **re-anchor**：

1. **Drop encoder cache**：保留最近 1 个 window，丢弃其余。`enc_cache_base_windows += drop_count`
2. **重置 decoder 状态**：清空 KV cache，携带最近 24 个 token 作为 prefix
3. **设置 `audio_trim_request`**：请求从 audio buffer 前端移除旧音频

Audio trim 在 `stream_push_audio` 返回后由 C API 层执行：

```rust
// c_api.rs
s.audio_buf.drain(..trim);           // 从 buffer 前端移除 samples
s.state.apply_audio_trim(trim);       // 调整 state 中的偏移量
```

### 2.3 两步操作的时序

```
                 stream_push_audio()
                 ┌─────────────────────────────────────────┐
                 │ chunk 7:  re-anchor triggered            │
                 │   → enc_cache: drop 1, keep 1            │
                 │   → enc_cache_base_windows: 0 → 1        │
                 │   → audio_trim_request = 128000          │
                 │ chunk 8:  正常处理（trim 尚未执行）       │
                 │ chunk 9:  正常处理                        │
                 │ ···                                       │
                 └─────────────────────────────────────────┘
                              ↓ 返回
                 C API 执行 audio trim:
                   audio_buf.drain(..128000)
                   apply_audio_trim(128000)
                              ↓
                 下一次 stream_push_audio()
                   此时 buffer 已经被截短
```

---

## 3. Bug 详情

### 3.1 Bug 所在

`apply_audio_trim` 在移除 audio buffer 前端的 samples 后，仅更新了 `audio_cursor` 和 `last_partial_cursor`，**没有更新 `enc_cache_base_windows`**：

```rust
// 修复前
pub fn apply_audio_trim(&mut self, trim: usize) {
    self.audio_cursor = self.audio_cursor.saturating_sub(trim);
    self.last_partial_cursor = self.last_partial_cursor.saturating_sub(trim);
    // ❌ enc_cache_base_windows 未更新
}
```

### 3.2 为什么这是致命的

audio trim 从 buffer **前端**移除 samples，使得所有绝对偏移失效。但 encoder window 偏移的计算公式：

```rust
let ws = (enc_cache_base_windows + enc_cache.len()) * enc_window_samples;
```

仍然基于 **原始 buffer 起点**计算。trim 之后，这个偏移远超实际 buffer 范围，导致 **永远无法编码新的 encoder window**。

### 3.3 修复方式

```rust
// 修复后
pub fn apply_audio_trim(&mut self, trim: usize) {
    self.audio_cursor = self.audio_cursor.saturating_sub(trim);
    self.last_partial_cursor = self.last_partial_cursor.saturating_sub(trim);
    // ✅ 同步更新 enc_cache_base_windows
    if self.enc_window_samples > 0 {
        let trimmed_windows = trim / self.enc_window_samples;
        self.enc_cache_base_windows =
            self.enc_cache_base_windows.saturating_sub(trimmed_windows);
    }
}
```

---

## 4. 数值推演示例

以 `long_60s_01` 的实际运行数据为例（`enc_window_samples = 128000`，`chunk_samples = 32000`）。

### 4.1 修复前：偏移错位 → encoder 冻结

**Chunk 7 (re-anchor 触发)**

```
状态：
  audio_cursor     = 256000   (处理了 8 个 chunk × 32000)
  enc_cache        = [win0, win1]  (2 个 window, 各 ~104 tokens)
  enc_cache_base   = 0

re-anchor 操作：
  drop 1 window, keep 1 → enc_cache = [win1]
  enc_cache_base   = 0 + 1 = 1
  audio_trim_request = 256000 - 128000 = 128000
```

**同一次 push 内的后续 chunk (8~11) 照常处理**——因为 trim 尚未执行，buffer 仍完整。

**push 返回后，C API 执行 trim**

```
audio_buf.drain(..128000)    → buffer 前 128000 samples 被移除
apply_audio_trim(128000):
  audio_cursor:  256000 → 128000  ✅
  enc_cache_base: 1 → 1          ❌ 未更新！
```

**下一次 push (chunk 12+)**

```
新 audio_cursor = 128000 + 32000 = 160000
full_end = (160000 / 128000) × 128000 = 128000

需要编码新 window 的条件：
  (enc_cache_base + enc_cache.len()) × enc_window_samples < full_end
= (1 + 1) × 128000 < 128000
= 256000 < 128000
= false   ❌ 不编码新 window！

此后每个 chunk, audio_cursor 增加 32000:
  chunk 12: 160000 → 256000 < 192000? no
  chunk 13: 192000 → 256000 < 192000? no
  chunk 14: 224000 → 256000 < 256000? no
  chunk 15: 256000 → 256000 < 256000? no
  ...直到 chunk 20: 384000 → (1+1)×128000 = 256000 < 384000? 是的
  但此时 samples[256000..384000] 在 trim 后的 buffer 中实际对应的是原始
  audio 的 384000~512000 偏移——window 内容已经错位！
```

**后果**：re-anchor 后长达 **8~12 个 chunk（16~24 秒）** encoder 不产出新 window。decoder 只看到单个旧 window 的 encoder 输出，无法生成有意义的新内容。触发 degeneracy reset → 同样的 carry tokens → 同样的冻结 → 无限循环。

实际 debug 日志佐证（修复前 chunk 16~37 的 encoder 行为）：

```
  [stream chunk 16] encoder: 50ms, prefill: 0/164 reused
  [stream chunk 17] encoder: 0ms,  prefill: 163/163 reused  ← encoder 冻结
  [stream chunk 18] encoder: 95ms, prefill: 113/215 reused
  [stream chunk 19] encoder: 0ms,  prefill: 113/137 reused  ← 始终 ~113 tokens
  [stream chunk 20] encoder: 42ms, prefill: 113/163 reused
  [stream degen] reset at chunk 20 (stale=4)                ← 退化重置
  ...完全相同的模式重复...
  [stream degen] reset at chunk 25 (stale=4)                ← 再次退化
  ...完全相同的模式重复...
  [stream degen] reset at chunk 37 (stale=4)                ← 再次退化
  FINAL RESULT: ...持续集成和持续   ← 截断，只有前 15 秒内容
```

从 chunk 16 到 chunk 37（**22 个 chunk，约 44 秒音频**），zero emit。

### 4.2 修复后：偏移正确 → encoder 持续增长

**同样的 Chunk 7 (re-anchor)**

```
re-anchor 操作（同上）：
  enc_cache_base = 0 → 1
  audio_trim_request = 128000
```

**push 返回后，C API 执行 trim**

```
apply_audio_trim(128000):
  audio_cursor:  256000 → 128000  ✅
  enc_cache_base: 1 → 1 - (128000/128000) = 1 - 1 = 0  ✅ 正确更新！
```

**下一次 push (chunk 12+)**

```
audio_cursor = 128000 + 32000 = 160000
full_end = 128000

需要编码新 window 的条件：
  (enc_cache_base + enc_cache.len()) × enc_window_samples < full_end
= (0 + 1) × 128000 < 128000
= 128000 < 128000
= false   但这是正确的——当前 window 已在 cache 中

audio_cursor 继续增长到 256000 时:
  full_end = 256000
  (0 + 1) × 128000 = 128000 < 256000
= true   ✅ 编码新 window!
  ws = (0 + 1) × 128000 = 128000
  samples[128000..256000]   ← 正确指向 trim 后 buffer 中的数据
```

**后果**：encoder 每 4 个 chunk（8 秒）产出一个新 window，decoder 持续看到更新的 encoder 输出，正常推进转录。

实际 debug 日志佐证（修复后）：

```
  [stream chunk  8] encoder: 45ms,  prefill: 0/164 reused    ← re-anchor 后第一个 chunk
  [stream chunk 10] encoder: 99ms,  prefill: 113/230 reused  ← 新 window! 113→230
  [stream chunk 11] encoder: 129ms, prefill: 113/264 reused  ← 继续增长
  [stream reanchor] at chunk 11 (enc_seq=208)                 ← 正常 re-anchor
  [stream chunk 12] encoder: 43ms,  prefill: 0/164 reused
  [stream chunk 14] encoder: 98ms,  prefill: 113/231 reused  ← 新 window! 又在增长
  [stream reanchor] at chunk 15 (enc_seq=208)
  ... 每 4 个 chunk 周期性 re-anchor，encoder 持续产出新 window ...
  [stream chunk 36] encoder: 45ms,  prefill: 0/164 reused
  [finalize] delta: "等多个层面综合考虑。"                    ← 完整输出到结尾
```

---

## 5. 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `crates/qwen-asr/src/transcribe.rs` | `StreamState` 新增 `enc_window_samples` 字段；`apply_audio_trim` 中同步更新 `enc_cache_base_windows` |

### 5.1 StreamState 新增字段

```rust
pub struct StreamState {
    // ...existing fields...

    /// Encoder window size in samples (set on first stream_push_audio call).
    /// Needed by apply_audio_trim to adjust enc_cache_base_windows.
    enc_window_samples: usize,
}
```

在 `stream_push_audio` 入口处赋值：

```rust
let enc_window_samples = enc_window_frames * HOP_LENGTH;
state.enc_window_samples = enc_window_samples;
```

### 5.2 apply_audio_trim 修复

```rust
pub fn apply_audio_trim(&mut self, trim: usize) {
    self.audio_cursor = self.audio_cursor.saturating_sub(trim);
    self.last_partial_cursor = self.last_partial_cursor.saturating_sub(trim);
    // 同步更新 enc_cache_base_windows
    if self.enc_window_samples > 0 {
        let trimmed_windows = trim / self.enc_window_samples;
        self.enc_cache_base_windows =
            self.enc_cache_base_windows.saturating_sub(trimmed_windows);
    }
}
```

---

## 6. 为什么这个 Bug 之前没有暴露

1. **短音频（<15 秒）不触发 re-anchor**，不会执行 audio trim，因此完全不受影响
2. **离线模式**使用 `transcribe_audio`/`transcribe_segment`，不涉及 audio trim
3. **早期开发中** re-anchor 后保留了所有 encoder window（`e9b8eb0` commit），不执行 audio trim，掩盖了这个问题
4. **流式长音频测试**之前没有逐 token 级别的 debug 日志，难以定位 encoder 冻结的根因

---

## 7. Benchmark 结果

修复后全部 67 条测试用例的流式 CER 与 MLX 参考实现基本对齐，短音频无退化。

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 流式平均 CER | 0.0631 | 0.0486 |
| long_60s_01 流式 CER | 0.707 | 0.023 |
| long_30s_01 流式 CER | 0.536 | 0.015 |
| CER=0 的条数（流式） | 35/67 | 35/67 |
| CER>0.20 的条数（流式） | 6 | 4 |
