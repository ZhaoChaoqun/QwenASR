# test_chunk_alignment.py — chunk_sec 与 encoder window 对齐假设验证

## 目的

验证一个关键假设：**chunk_sec 能否整除 encoder window（8.0s）是长音频 CER 退化的决定性因素**。

`param_isolation_test.py` 的渐变测试发现 CER 随 chunk_sec 减小呈非单调退化（1.8s 好于 1.9s 和 1.7s），提示可能存在对齐效应。本脚本通过系统性地测试整除与非整除两组 chunk_sec 值来验证这一假设。

## 背景

QwenASR 流式引擎的 encoder 以固定大小的 window 处理音频：

```
encoder window = enc_n_window_infer(800) × HOP_LENGTH(160) = 128,000 samples = 8.0s
```

当 `chunk_sec × 16000` 不能被 128,000 整除时，`audio_cursor` 按 chunk_samples 步进会停在非 window 边界处，导致 partial encoder 反复编码不完整的 window 尾部。

## 验证方法

### 测试假设

- **整除组**（chunk_samples 整除 128,000）：应无退化
- **非整除组**（chunk_samples 不整除 128,000）：应退化

### 测试配置

| # | chunk_sec | chunk_samples | 128000 / chunk_samples | 组别 |
|---|:---------:|:-------------:|:----------------------:|------|
| A | 2.0 | 32,000 | 4.00 | 整除（基线） |
| B | 1.0 | 16,000 | 8.00 | 整除 |
| C | 0.5 | 8,000 | 16.00 | 整除 |
| D | 1.6 | 25,600 | 5.00 | 整除 |
| E | 4.0 | 64,000 | 2.00 | 整除 |
| F | 1.5 | 24,000 | 5.33 | 非整除 |
| G | 1.7 | 27,200 | 4.71 | 非整除 |
| H | 1.9 | 30,400 | 4.21 | 非整除 |
| I | 1.8 | 28,800 | 4.44 | 非整除 |

其他参数固定为 Rust 默认值（`rollback=5, unfixed_chunks=2, max_new_tokens=32`）。

### 测试数据

同 `param_isolation_test.py`，3 条长音频：`zh_long_01`（9s）、`long_30s_01`（43s）、`long_60s_01`（75s）。

## 结论与发现

### 1. 对齐是重要因素，但不是唯一因素

测试结果显示：

- **非整除组全部退化**——1.5s、1.7s、1.8s、1.9s 在长音频上均有不同程度的 CER 升高，验证了对齐假设的部分成立
- **整除组中小 chunk_sec 也退化**——chunk_sec=0.5s 和 1.0s 虽然整除 encoder window，但因 chunk 过小（单次送入的新音频太短），encoder 上下文信息不足，同样产生退化

因此结论是：**对齐是必要条件但非充分条件**。除了对齐，chunk_sec 还需要足够大才能保证 encoder 有充足的上下文信息。

### 2. chunk_sec=2.0 是当前最优值

在所有测试值中，只有 chunk_sec=2.0 同时满足两个条件：
1. **整除 encoder window**（128,000 / 32,000 = 4.0）
2. **chunk 足够大**（2.0s 提供充足的新音频上下文）

chunk_sec=4.0 虽然也整除且更大，但增大了首字延迟（需累积 4s 音频才出第一个结果）。

### 3. 非单调退化行为的解释

`param_isolation_test.py` 中观察到的 chunk_sec=1.8 优于 1.7 和 1.9 的非单调现象：

```
1.8s → 28,800 samples → 128,000 / 28,800 ≈ 4.44
1.7s → 27,200 samples → 128,000 / 27,200 ≈ 4.71
1.9s → 30,400 samples → 128,000 / 30,400 ≈ 4.21
```

不同 chunk_sec 导致 partial encoder 尾部长度各异。1.8s 的余数恰好使 partial 编码片段略长，encoder 得到稍多上下文，质量略好。这属于 encoder 对 partial 输入的数值敏感性，不具备可预测性。

### 4. 不建议追求对齐优化

虽然对齐会影响 CER，但试图寻找满足对齐的更小 chunk_sec（如 1.6s）带来的首字延迟收益有限（仅 0.4s），而引入的质量风险较高。最可靠的方案仍是保持 `chunk_sec=2.0`。

## 依赖

```bash
uv run scripts/test_chunk_alignment.py
```

需要：
- `~/Github/typeless/Frameworks/qwen-asr/lib/libqwen_asr.dylib`
- Qwen3-ASR-0.6B 模型（HuggingFace cache）
- 测试音频集（`typeless/Tests/fixtures/`）
