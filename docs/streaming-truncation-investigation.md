# 流式长音频截断问题调查报告

*调查时间：2026-03-03*

## 背景

Swift FFI benchmark 显示流式模式在长音频上有严重截断：`long_60s_01` CER=0.654，`long_30s_01` CER=0.567。
而 Rust benchmark 报告中流式 CER 与离线一致（均为 0.0576），疑似测试方法有问题。

本报告记录调查发现的所有问题及其根因分析。

## 涉及的代码路径

系统中存在**三条**流式相关代码路径，理解它们的区别是定位问题的关键：

| 路径 | 入口 | 调用者 | 说明 |
|------|------|--------|------|
| A. `transcribe_stream()` | `transcribe.rs:447` | Rust CLI `--stream` | 一次性拿到全部音频，内部按 chunk 切分处理 |
| B. `stream_push_audio()` | `transcribe.rs:941` | C API `qwen_asr_stream_push()` → Swift FFI | 增量推送音频，持久化 `StreamState` |
| C. `transcribe_segment()` | `transcribe.rs` | `transcribe_stream()` 无 callback 时 fallback | 标准离线分段解码 |

Swift FFI 走的是路径 B，Rust CLI `--stream --silent` 走的是路径 A→C（fallback 到离线）。

---

## 问题列表

### Issue 1 [P0] — `stream_push_audio` 的 `speech_ended` 提前终止，丢弃后续全部音频

**文件**: `crates/qwen-asr/src/transcribe.rs`

**问题代码**:

```rust
// L1190-1192 — 检测 speech end
let speech_ended = chunk_tokens.is_empty()         // decoder 产生了 EOT
    && !state.raw_tokens.is_empty()                 // 已有识别结果
    && state.chunk_idx >= unfixed_chunks;            // 已过 unfixed 阶段

// L1194-1220 — speech_ended 时 emit 剩余 rollback tokens（正确）

// L1354-1356 — 直接 break 跳出 while 循环（问题！）
if speech_ended {
    break;
}
```

**触发场景**: 长音频中间出现较长 silence 间隙时，re-anchor 重置 decoder 状态后，decoder 对 silence chunk 产生 EOT（`chunk_tokens` 为空）。此时 `speech_ended=true`，while 循环 break，**后续所有未处理的音频 chunk 全部丢弃**。

**实际表现** (`long_60s_01` Swift 流式输出):
```
软件工程是一门研究用工程化方法构建和维护有效的、实用的和高质量的软件的学科。
它涉及到程序设计语言、方法，强调快速迭代。持续集成审查是保证代码质量集成测试
帮助开发者在的小型服务。容器云计算平台提供了弹性的软件的快速软件的快速可靠交付
层面综合考虑。
```

中间大段文本被跳过，且出现拼接碎片（多次 re-anchor 后 carry tokens 的残留）。

**对比**: `transcribe_stream()`（路径 A）**没有 `speech_ended` 检测**，所有 chunk 都会处理到结尾，因此不会截断。

**根因**: `stream_push_audio` 设计时假设"speech ended = 用户停止说话 = 可以结束"，但在文件输入或长音频场景下，中间 silence 不代表全部音频结束。而 `transcribe_stream` 因为预先知道全部音频长度，用 `is_final = audio_cursor >= audio_samples.len()` 判断，不依赖 decoder EOT。

**修复方向**: `speech_ended` 时不应 break 整个 while 循环。应像 re-anchor 一样做 carry token 保存 + 状态重置，然后继续处理后续音频。只在外部调用者传入 `finalize=true` 且音频确实处理完毕时才真正结束。

**状态**: 🔴 待修复

---

### Issue 2 [P0] — Rust benchmark 的"流式" CER 是假象，实际走的是离线解码

**文件**:
- `crates/qwen-asr-cli/src/main.rs` L198, L260, L390, L411-418
- `crates/qwen-asr/src/transcribe.rs` L465-472
- `scripts/benchmark.py` L74

**完整调用链分析**:

`scripts/benchmark.py` 使用以下命令运行流式 benchmark：

```python
# benchmark.py L74
cmd = [str(BINARY), "-d", str(MODEL_DIR), "-i", wav_path, "--silent"]
if stream:
    cmd.append("--stream")
```

即 `qwen-asr -d <model> -i <wav> --silent --stream`。然而在 Rust CLI 中：

```
步骤 1: --silent → verbosity = 0
         main.rs:198  "-s" | "--silent" => verbosity = 0

步骤 2: verbosity = 0 → emit_tokens = false
         main.rs:260  let emit_tokens = verbosity > 0;

步骤 3: emit_tokens = false → token_cb 不设置
         main.rs:390  if emit_tokens {
         main.rs:391      ctx.token_cb = Some(Box::new(stream_token));
         main.rs:392  }
                      // emit_tokens=false 时这段代码不执行，token_cb 保持 None

步骤 4: --stream → 调用 transcribe_stream()
         main.rs:411  let text = if stream_mode {
         main.rs:418      Some(s) => transcribe::transcribe_stream(&mut ctx, &s),

步骤 5: transcribe_stream() 检测到 token_cb 为 None → fallback 到离线解码
         transcribe.rs:464-472
         if ctx.token_cb.is_none() {
             // ...
             let (text, _) = transcribe_segment(ctx, &audio_samples, &tokenizer, None)?;
             return Some(text);       // ← 实际执行的是离线 transcribe_segment！
         }
```

**结果**: Rust benchmark 报告中"Qwen3-ASR (流式)" 的 CER=0.0576，与离线 CER=0.0576 完全一致——因为它们执行的是**同一段代码**。报告数据具有误导性，掩盖了真实流式路径（`stream_push_audio`）的质量问题。

**修复方向**: 两个可选方案：
- (a) 让 benchmark 使用 `stream_push_audio` API（模拟 C API 调用），获取真正的流式 CER
- (b) 让 `transcribe_stream` 在无 `token_cb` 时也走真正的流式路径（而非 fallback 到离线）

推荐方案 (a)，因为这更贴近 Swift FFI 的实际调用方式。

**状态**: ✅ 已修复 — `scripts/benchmark.py` 已改为通过 ctypes 调用 `libqwen_asr.dylib` C API，离线使用 `qwen_asr_transcribe_pcm()`，流式使用 `qwen_asr_stream_push()` + 2s chunk + 0.1s silence finalize（与 Swift benchmark 同一代码路径）。验证结果：`long_60s_01` 流式 CER=0.871（确认了截断问题存在），离线 CER=0.009。

---

### Issue 3 [P1] — Swift 流式参数比默认值更激进，加剧截断

**文件**: `typeless/Sources/QwenASRRecognizer.swift` L30-35

**当前设置 vs 默认值**:

```swift
// QwenASRRecognizer.swift L30-35
qwen_asr_stream_set_chunk_sec(engine, 1.5)       // 默认 2.0
qwen_asr_stream_set_rollback(engine, 3)           // 默认 5
qwen_asr_stream_set_unfixed_chunks(engine, 1)     // 默认 2
qwen_asr_stream_set_max_new_tokens(engine, 32)    // 默认 32（一致）

// Rust 默认值定义在 context.rs L248-251:
// stream_chunk_sec: 2.0,
// stream_rollback: 5,
// stream_unfixed_chunks: 2,
// stream_max_new_tokens: 32,
```

**各参数的具体影响**:

| 参数 | Rust 默认 | Swift 设置 | 具体影响 |
|------|:---------:|:----------:|----------|
| `chunk_sec` | 2.0 | **1.5** | chunk 更小（24000 vs 32000 samples），re-anchor 触发更频繁。`STREAM_RESET_INTERVAL_CHUNKS=45` 对应 67.5s vs 90s |
| `rollback` | 5 | **3** | 回滚窗口从 5 tokens 缩小到 3 tokens，纠错空间减少 40% |
| `unfixed_chunks` | 2 | **1** | `speech_ended` 条件中 `chunk_idx >= unfixed_chunks` 从第 3 个 chunk 起生效变为第 2 个 chunk 起生效，speech_ended 更早触发 |

**与 Issue 1 的交互**: `unfixed_chunks=1` 使得 `speech_ended` 的触发条件 `state.chunk_idx >= unfixed_chunks` 在第 2 个 chunk 就满足。在 re-anchor 重置后，如果第一个 chunk 是 silence，decoder 产生 EOT，就立刻 speech_ended → break，大段音频丢失。如果 `unfixed_chunks=2`，则第一个 chunk 即使 EOT 也不会触发。

**修复方向**: 在 Issue 1 修复前，考虑将参数恢复为默认值以缓解截断。Issue 1 修复后重新评估最优参数组合。

**状态**: 🟡 待评估

---

### Issue 4 [P2] — `stream_push_audio` 与 `transcribe_stream` 的 degeneracy reset 行为不一致

**文件**: `crates/qwen-asr/src/transcribe.rs`

**差异对比**:

| 场景 | `transcribe_stream` (路径 A) | `stream_push_audio` (路径 B) |
|------|:-------------------:|:-------------------:|
| degeneracy reset 保留 window | 2 个 (L711) | **1** 个 (L1260) |
| re-anchor 保留 window | 1 个 (L749) | 1 个 (L1300) |

```rust
// transcribe_stream degeneracy reset — L711
let keep_windows = 2.min(enc_cache.len());    // 保留 2 个 window

// stream_push_audio degeneracy reset — L1260
let keep_windows = 1.min(enc_cache.len());    // 保留 1 个 window
```

每个 encoder window 约 8s 音频上下文。`stream_push_audio` 在 degeneracy reset 后仅保留 ~8s 上下文（vs `transcribe_stream` 的 ~16s），decoder 可用信息更少，更容易再次 degenerate 或产生 EOT。

**修复方向**: 统一两个函数的 degeneracy reset 行为。考虑到最近的 commit `8978471` ("fix: revert to keep 1 encoder window, add degen enc cache trim") 将 re-anchor 统一为 1 window，degeneracy reset 也应该保持一致（都用 1）。但需要通过测试验证。

**状态**: 🟡 待评估

---

## 截断实际表现

### Swift Benchmark 日志 (2026-03-03)

**`long_60s_01`** (CER=0.654):

期望文本（312 字，12 个句子）：
> 软件工程是一门研究用工程化方法构建和维护有效的实用的和高质量的软件的学科。它涉及到程序设计语言、数据库、软件开发工具、系统平台等方面的知识。现代软件开发通常采用敏捷开发方法，强调快速迭代和持续交付。版本控制系统如Git是团队协作开发的基础工具。持续集成和持续部署能够自动化测试和发布流程，提高开发效率。代码审查是保证代码质量的重要实践，团队成员互相审阅代码修改。单元测试和集成测试帮助开发者在早期发现和修复缺陷。微服务架构将大型应用拆分为多个独立的小型服务。容器化技术如Docker简化了应用的部署和运维管理。云计算平台提供了弹性的计算资源，支持应用的快速扩展。DevOps实践将开发和运维紧密结合，促进软件的快速可靠交付。性能优化需要从算法、数据结构、系统架构等多个层面综合考虑。

实际流式输出（严重截断 + 碎片拼接）：
> 软件工程是一门研究用工程化方法构建和维护有效的、实用的和高质量的软件的学科。它涉及到程序设计语言、方法，强调快速迭代。持续集成审查是保证代码质量集成测试帮助开发者在的小型服务。容器云计算平台提供了弹性的软件的快速软件的快速可靠交付层面综合考虑。

丢失了约 2/3 的内容，中间出现了多处 carry token 拼接碎片。

**`long_30s_01`** (CER=0.567):

期望文本（8 个句子）：
> 人工智能技术在过去10年中取得了巨大的进步。深度学习算法使得计算机能够处理和理解自然语言。语音识别技术已经广泛应用于智能手机和智能音箱。自动驾驶汽车使用多种传感器和人工智能算法来感知环境。医疗领域的人工智能可以辅助医生进行疾病诊断。自然语言处理技术让机器能够理解人类的语言并做出回应。计算机视觉技术使得机器能够识别和分析图像中的内容。强化学习技术让人工智能系统能够通过试错来学习最优策略。

实际流式输出：
> 人工智能技术在过去十年中取得了巨大的进步。深度学习算法使得计算机能够处理和理解自然语言、语音识别技术已经广泛来感知环境。医疗语言处理技术让识别和分析图像中的来学习最优策略。

同样丢失约 2/3 内容。

### 对比：离线模式同一音频

| 条目 | 离线 CER | 流式 CER | 差距 |
|------|:--------:|:--------:|:----:|
| `long_60s_01` | 0.009 | 0.654 | 72x |
| `long_30s_01` | 0.010 | 0.567 | 57x |

离线模式基本完美，说明模型本身没有问题，问题完全出在流式路径的状态管理上。

---

## 修复优先级

1. **Issue 1 [P0]** — 修复 `speech_ended` break 逻辑（截断的直接原因）
2. **Issue 2 [P0]** — 修复 Rust benchmark 使其测试真正的流式路径（否则无法验证 Issue 1 的修复效果）
3. **Issue 3 [P1]** — 评估并调整 Swift 流式参数（Issue 1 修复后重新评估）
4. **Issue 4 [P2]** — 统一 degeneracy reset window 保留策略

## 验证方式

修复后运行以下测试验证：

```bash
# Rust 真实流式 benchmark（Issue 2 修复后）
uv run scripts/benchmark.py

# Swift benchmark
cd ~/Github/typeless
xcodebuild test -scheme Typeless -destination 'platform=macOS' \
  -only-testing:TypelessTests/ASRPipelineBenchmarkTests/testQwenASRStreamPipeline
```

重点关注条目：
- `long_60s_01`：期望流式 CER < 0.05（当前 Swift 0.654）
- `long_30s_01`：期望流式 CER < 0.05（当前 Swift 0.567）
- 短音频 CER 不应退化（当前短音频流式质量良好）
