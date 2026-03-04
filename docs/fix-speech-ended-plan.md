# 修复 Issue 1 [P0]: `speech_ended` break 导致长音频截断

*计划时间：2026-03-03*

## Context

`stream_push_audio()` 中，当 decoder 产生 EOT（`chunk_tokens` 为空）时，`speech_ended=true` 导致 while 循环 `break`，后续所有未处理的音频被丢弃。这是长音频截断的直接原因（long_60s_01 流式 CER=0.871 vs 离线 CER=0.009）。

参考 MLX Python 实现：decoder EOT 不终止处理循环，只在 `finalize && audio_cursor >= len(audio_buffer)` 时才退出。

---

## 修改文件

| 文件 | 修改内容 |
|------|----------|
| `crates/qwen-asr/src/transcribe.rs` | `stream_push_audio()` 中完全删除 `speech_ended` 逻辑，增加 audio trim |
| `crates/qwen-asr/src/c_api.rs` | `qwen_asr_stream_push()` 中增加 audio buffer trim |

---

## 方案对比：完全删除 vs 保留检测改为 re-anchor

### 方案 A: 完全删除 `speech_ended`（推荐）

**做法**: 删除 L1186-1220 的 speech_ended 检测和 rollback emit，删除 L1227/L1312 的 `if !speech_ended` 条件守卫，删除 L1353-1356 的 break。所有 chunk（包括 decoder EOT 产生空 `chunk_tokens` 的情况）统一走 degeneracy → reanchor → stabilization 流程。

**decoder EOT 时的行为**:
1. `chunk_tokens` 为空
2. L1223: `raw_tokens.truncate(n_prefix_tokens)` → 只保留 prefix
3. L1224: `raw_tokens.extend_from_slice(&[])` → 无新增
4. degeneracy 检测：`raw_tokens == prev_tail_snapshot` → stale_count 递增
5. stabilization：candidate_len 不变 → 不 emit 新 token
6. `chunk_idx += 1`，继续下一个 chunk
7. 连续 4 个 stale chunk → degeneracy 触发 → re-anchor 式重置

**优点**:
- 代码最简洁，删除 ~40 行特殊分支
- 与 MLX 行为完全一致
- rollback tokens 不需要特殊 flush — 后续有语音的 chunk 会通过正常 stabilization 自然 emit 它们
- 不引入新的状态重置路径，降低 bug 风险

**缺点**:
- decoder EOT 后的 rollback tokens 不会立即 emit，而是延迟到后续有语音 chunk 时 emit（通过 stabilization 的 monotonic commit）
- 如果 EOT 后的剩余音频全是 silence 直到 finalize，rollback tokens 会在 finalize 时 flush（`is_final` 时 `candidate_len = n_text_tokens`）

**延迟 emit 是否有问题？** 不是。流式 ASR 的 rollback 窗口本身就是"可能会变"的 speculative tokens。延迟几个 chunk emit 不影响最终文本正确性。且在实际使用中（麦克风输入），silence 后通常紧跟用户的下一句话，延迟几乎不可察觉。

---

### 方案 B: 保留检测，改为 re-anchor 式重置

**做法**: 保留 L1190-1220 的 speech_ended 检测和 rollback emit，在 emit 后加入 re-anchor 重置逻辑（carry tokens + clear caches），然后删除 break 让循环继续。

**优点**:
- rollback tokens 在 speech_ended 时立即 emit（更低延迟）
- 保留了显式的"speech ended"语义，便于日后扩展（如实时场景的 utterance 边界检测）

**缺点**:
- 增加一个新的状态重置路径（speech_ended reset），与 degeneracy reset、re-anchor reset 是**第三个**几乎相同但略有差异的重置逻辑，增维护负担
- rollback emit 的代码与 stabilization emit 的代码功能重叠，但逻辑不同（前者 emit 全部剩余 token，后者 emit 到 candidate_len）
- `if !speech_ended` 条件守卫仍需保留，代码分支更复杂

---

### 推荐：方案 A

方案 A 更简洁，也与 MLX 的行为完全一致。rollback tokens 的延迟 emit 在实践中不会造成问题，因为 stabilization 的 monotonic commit 机制保证了正确性。

---

## 修改内容

### 修改 1: 完全删除 `speech_ended` 逻辑

**位置**: `transcribe.rs` L1186-1356

**删除以下代码块**:

**(a)** L1186-1220: `speech_ended` 检测 + rollback emit 分支（整块删除）

```rust
// 删除 ↓
let speech_ended = chunk_tokens.is_empty()
    && !state.raw_tokens.is_empty()
    && state.chunk_idx >= unfixed_chunks;

if speech_ended {
    // ... entire rollback emit block ...
}
// 删除 ↑
```

**(b)** L1227 和 L1312: 移除 `if !speech_ended` 条件守卫（保留内部逻辑）

```rust
// 之前:
if !speech_ended {
    // degeneracy + reanchor logic
}
// ...
if !speech_ended {
    // stabilization emit logic
}

// 之后: 直接执行，不需要条件守卫
// degeneracy + reanchor logic
// ...
// stabilization emit logic
```

**(c)** L1353-1356: 删除 `if speech_ended { break; }`

---

### 修改 2: 在 re-anchor/speech_ended 重置后裁剪 audio buffer

**问题**: 即使去掉 `speech_ended` break，re-anchor 后 `audio_cursor` 不调整，encoder 序列长度在 re-anchor 后可能立即再次超过阈值，导致频繁触发 re-anchor 循环。

**方案**: 在 `StreamState` 中增加 `audio_trim_request` 字段，记录可以从 audio buffer 头部裁掉的 sample 数。由 `c_api.rs` 层在 `stream_push_audio` 返回后执行实际的 buffer 裁剪。

#### 具体修改

**(a)** `StreamState` 增加字段和方法：

```rust
// 新字段
pub audio_trim_request: usize,

// 新方法
pub fn apply_audio_trim(&mut self, trim: usize) {
    self.audio_cursor = self.audio_cursor.saturating_sub(trim);
    self.last_partial_cursor = self.last_partial_cursor.saturating_sub(trim);
}
```

**(b)** 在 re-anchor/speech_ended/degeneracy 重置时（3 个位置）设置 trim：

```rust
// 保留最近 ~8s audio context（与 MLX 的 chunk_samples * 4 一致）
let keep_samples = chunk_samples * 4;
if state.audio_cursor > keep_samples {
    state.audio_trim_request = state.audio_cursor - keep_samples;
}
```

**(c)** `c_api.rs` 中，调用 `stream_push_audio` 后处理 trim：

```rust
// 在 qwen_asr_stream_push() 中，stream_push_audio 调用后
if s.state.audio_trim_request > 0 {
    let trim = s.state.audio_trim_request.min(s.audio_buf.len());
    s.audio_buf.drain(..trim);
    s.state.apply_audio_trim(trim);
    s.state.audio_trim_request = 0;
}
```

---

### 修改 3: 增大关键参数（与 MLX 对齐）

| 参数 | 当前值 | 新值 | 来源 |
|------|:------:|:----:|------|
| `STREAM_REANCHOR_ENC_SEQ_THRESHOLD` | 200 (~15s) | **400** (~30s) | `transcribe.rs:31` |
| `STREAM_RESET_INTERVAL_CHUNKS` | 45 (~90s) | **90** (~180s) | `transcribe.rs:20` |
| `stream_max_new_tokens` 默认值 | 32 | **128** | `context.rs` |

---

## 不修改的内容

- byte-level alignment（P4，后续单独处理）
- KV cache 策略（Moderate，不影响截断）
- `transcribe_stream()` 路径（不影响 C API / Swift FFI）

---

## 验证

```bash
# 构建
cargo build --release -p qwen-asr

# 拷贝 dylib 到 typeless（如需 Swift 测试）
cp target/release/libqwen_asr.dylib ~/Github/typeless/Frameworks/

# 运行 Rust benchmark（通过 ctypes C API）
uv run scripts/benchmark.py

# 关注指标：
# - long_60s_01 流式 CER: 期望 < 0.05（当前 0.871）
# - long_30s_01 流式 CER: 期望 < 0.05（当前 0.567）
# - 短音频 CER 不应退化
```

---

## 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| decoder EOT 后 rollback tokens 延迟 emit | 非实时场景无影响；实时场景延迟几个 chunk (~2-4s) | finalize 时全部 flush；后续有语音 chunk 时通过 stabilization 自然 emit |
| 连续 silence chunks 的 CPU 开销 | encoder + decoder 空跑（encoder 有缓存命中，decoder 很快 EOT） | degeneracy 检测在 4 个 stale chunk 后触发 re-anchor，限制空跑 |
| max_new_tokens=128 增大推理延迟 | 单 chunk 最多 4x decode | 实际中文语音很少超过 64 tokens/2s，多数 chunk 提前 EOT |
| audio trim 偏移量计算错误导致 encoder 看到错误音频段 | 严重质量退化 | trim 逻辑简单（仅 audio_cursor - keep_samples），且只影响 re-anchor 后 |
