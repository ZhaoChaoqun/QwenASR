# 流式 chunk_sec 参数导致长音频 CER 退化分析报告

*生成时间：2026-03-04*
*测试环境：Apple Silicon, dylib commit a38245a (含 set_language fix)*

---

## 1. 问题描述

Swift 产品代码为降低首字延迟，将 streaming 参数从 Rust 默认值调整为：

| 参数 | 默认值 | Swift 值 | 调整目的 |
|------|:------:|:--------:|----------|
| `chunk_sec` | 2.0s | **1.5s** | 缩小窗口，降低首字延迟 |
| `rollback` | 5 tokens | **3 tokens** | 减少回退，降低延迟 |
| `unfixed_chunks` | 2 chunks | **1 chunk** | 减少冷启动期 |
| `max_new_tokens` | 32 tokens | 32 tokens | 保持默认 |

调整后，长音频（>9s）的流式 CER 急剧恶化，从 ~0.01 升至 ~0.70。

---

## 2. 参数隔离测试

对 3 条长音频进行系统性 A/B 测试，每次仅改变一个参数，定位根因。

### 2.1 单参数隔离

| 配置 | chunk | rb | uf | zh_long (9s) | long_30s (43s) | long_60s (75s) | 平均 CER |
|------|:-----:|:--:|:--:|:---:|:---:|:---:|:---:|
| **A. 默认参数** | 2.0 | 5 | 2 | 0.000 | 0.010 | 0.009 | **0.006** |
| **C. 仅改 chunk_sec=1.5** | **1.5** | 5 | 2 | **0.537** | **0.665** | **0.912** | **0.705** |
| D. 仅改 rollback=3 | 2.0 | **3** | 2 | 0.000 | 0.015 | 0.012 | 0.009 |
| E. 仅改 unfixed_chunks=1 | 2.0 | 5 | **1** | 0.000 | 0.010 | 0.009 | 0.006 |

**结论：`chunk_sec=1.5` 是唯一导致退化的参数。** `rollback=3` 和 `unfixed_chunks=1` 对长音频 CER 无有意义的负面影响。

### 2.2 两参数组合

| 配置 | chunk | rb | uf | 平均 CER | 退化？ |
|------|:-----:|:--:|:--:|:---:|:---:|
| F. chunk=1.5 + rollback=3 | 1.5 | 3 | 2 | 0.702 | 是（跟随 chunk） |
| G. chunk=1.5 + unfixed=1 | 1.5 | 5 | 1 | 0.705 | 是（跟随 chunk） |
| **H. rollback=3 + unfixed=1** | **2.0** | **3** | **1** | **0.009** | **否** |

**结论：只要 `chunk_sec` 保持 2.0，其他参数任意调整均不影响长音频质量。**

### 2.3 chunk_sec 渐变测试

| chunk_sec | zh_long (9s) | long_30s (43s) | long_60s (75s) | 平均 CER |
|:---------:|:---:|:---:|:---:|:---:|
| 2.0 | 0.000 | 0.010 | 0.009 | 0.006 |
| 1.9 | 0.000 | 0.134 | 0.176 | 0.103 |
| 1.8 | 0.000 | 0.067 | 0.135 | 0.067 |
| 1.7 | 0.000 | 0.155 | 0.249 | 0.135 |
| 1.5 | 0.537 | 0.665 | 0.912 | 0.705 |

**结论：CER 退化从 chunk_sec < 2.0 时开始出现，随 chunk_sec 减小急剧恶化。1.5s 时已完全不可用。**

> 注：1.9 和 1.7 的 CER 比 1.8 更高，呈非单调关系。推测与 chunk 边界和 encoder window 对齐有关。

### 2.4 chunk=1.5 + 增大 rollback（无效）

| rollback | long_30s CER | long_60s CER |
|:--------:|:---:|:---:|
| 3 | 0.665 | 0.903 |
| 4 | 0.665 | 0.909 |
| 5 | 0.665 | 0.912 |
| 6 | 0.665 | 0.918 |
| 8 | 0.670 | 0.924 |

**增大 rollback 无法缓解 chunk_sec=1.5 的退化。** 问题不在 rollback 窗口大小。

---

## 3. 根因分析

### 3.1 关键参数关系

Rust streaming 引擎的核心参数：

```
encoder window = enc_n_window_infer(800) × HOP_LENGTH(160) = 128,000 samples = 8.0s
chunk_samples  = chunk_sec × 16,000
re-anchor 阈值 = STREAM_REANCHOR_ENC_SEQ_THRESHOLD = 200 encoder tokens ≈ 15s
```

### 3.2 chunk_sec=2.0 时的正常流程

- 每个 chunk = 32,000 samples (2.0s)
- encoder window = 128,000 samples (8.0s)
- 大约 4 个 chunk 填满一个 encoder window
- re-anchor 时（~15s），已累积约 2 个完整 encoder windows
- re-anchor 后保留 1 个 window（8s context），decoder 有足够的 encoder 上下文继续

### 3.3 chunk_sec=1.5 时的异常

- 每个 chunk = 24,000 samples (1.5s)
- encoder window = 128,000 samples (8.0s)
- 大约 5.3 个 chunk 填满一个 encoder window

关键问题在于 **chunk_samples 与 encoder window 的对齐**。`stream_push_audio` 中（`transcribe.rs:1002-1009`）：

```rust
state.audio_cursor = (state.audio_cursor + chunk_samples).min(samples.len());
```

`audio_cursor` 按 `chunk_samples`（24,000）步进，而 encoder window 编码按 `enc_window_samples`（128,000）对齐。当 chunk < window 时，每次 push 后 `audio_cursor` 停在非 window 边界处，partial encoder 被反复编码。

更关键的是，**partial encoder output 的质量在 chunk_sec 较小时不稳定**——每个 partial 只有 1.5s 的新音频，encoder 上下文碎片化，导致 encoder token 质量下降，进而影响 decoder。

### 3.4 退化模式观察

从测试输出可以看到两种退化模式：

1. **截断**（`long_60s_01` 只输出 31-34 字 vs 正常 346 字）：decoder 在 re-anchor 后很快生成了 EOS token，因为 encoder context 不足以支撑连续解码。

2. **重复后截断**（`zh_long_01` 从 9s 音频输出重复内容）：较短音频也受影响，说明 partial encoder 的碎片化编码本身就不稳定。

### 3.5 非单调 CER 行为的解释

chunk_sec=1.8 比 1.7 和 1.9 效果更好，这是因为：
- 1.8s × 16000 = 28,800 samples
- 128,000 / 28,800 ≈ 4.44 个 chunk
- 不同 chunk_sec 导致不同的 partial 尾部长度，恰好某些值在 window 边界处对齐更好

---

## 4. 建议修复方向

### 方案 A：Swift 端 — 保持 chunk_sec=2.0（推荐，立即可用）

将 `chunk_sec` 保持 2.0，仅调整 `rollback` 和 `unfixed_chunks` 来降低延迟：

```swift
qwen_asr_set_language(engine, "chinese")
qwen_asr_stream_set_chunk_sec(engine, 2.0)     // 保持默认
qwen_asr_stream_set_rollback(engine, 3)         // 降低延迟 ✓
qwen_asr_stream_set_unfixed_chunks(engine, 1)   // 降低延迟 ✓
qwen_asr_stream_set_max_new_tokens(engine, 32)
```

延迟影响估算：
- chunk_sec 2.0→1.5 降低 **0.5s** 首字延迟
- rollback 5→3 降低 **~0.3s** 延迟
- unfixed_chunks 2→1 降低 **~2.0s** 冷启动延迟

采用 `2.0/3/1/32` 可以获得 rollback + unfixed 的延迟收益（~2.3s），同时保持长音频稳定（CER=0.009）。

### 方案 B：Rust 端 — 修复 partial encoder 碎片化问题

需要在 `stream_push_audio` 中改进 partial encoder 的处理策略：

1. **当 chunk_sec < enc_window 时，保证 partial 编码的最小音频长度**：避免对过短的 partial 尾部单独编码
2. **re-anchor 时保留更多 encoder windows**：当 chunk_sec 较小时，单个 window 的 decoder context 可能不够
3. **将 re-anchor 阈值与 chunk_sec 关联**：使 re-anchor 周期与实际音频时长而非 encoder token 数量对齐

这些改动需要仔细设计和验证，不建议作为紧急修复。

---

## 5. 测试数据完整表

| # | Config | chunk | rb | uf | zh_long | long_30s | long_60s | AVG |
|---|--------|:-----:|:--:|:--:|:-------:|:--------:|:--------:|:---:|
| A | 默认参数 | 2.0 | 5 | 2 | 0.000 | 0.010 | 0.009 | 0.006 |
| B | Swift 全部自定义 | 1.5 | 3 | 1 | 0.537 | 0.665 | 0.903 | 0.702 |
| C | 仅改 chunk_sec=1.5 | 1.5 | 5 | 2 | 0.537 | 0.665 | 0.912 | 0.705 |
| D | 仅改 rollback=3 | 2.0 | 3 | 2 | 0.000 | 0.015 | 0.012 | 0.009 |
| E | 仅改 unfixed_chunks=1 | 2.0 | 5 | 1 | 0.000 | 0.010 | 0.009 | 0.006 |
| F | chunk=1.5 + rb=3 | 1.5 | 3 | 2 | 0.537 | 0.665 | 0.903 | 0.702 |
| G | chunk=1.5 + uf=1 | 1.5 | 5 | 1 | 0.537 | 0.665 | 0.912 | 0.705 |
| H | rb=3 + uf=1 | 2.0 | 3 | 1 | 0.000 | 0.015 | 0.012 | 0.009 |
| I | chunk_sec=1.7 | 1.7 | 5 | 2 | 0.000 | 0.155 | 0.249 | 0.135 |
| J | chunk_sec=1.8 | 1.8 | 5 | 2 | 0.000 | 0.067 | 0.135 | 0.067 |
| K | chunk_sec=1.9 | 1.9 | 5 | 2 | 0.000 | 0.134 | 0.176 | 0.103 |
| L | chunk=1.5 + rb=4 | 1.5 | 4 | 2 | 0.537 | 0.665 | 0.909 | 0.704 |
| M | chunk=1.5 + rb=6 | 1.5 | 6 | 2 | 0.805 | 0.665 | 0.918 | 0.796 |
| N | chunk=1.5 + rb=8 | 1.5 | 8 | 2 | 0.683 | 0.670 | 0.924 | 0.759 |
