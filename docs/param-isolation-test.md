# param_isolation_test.py — 流式参数隔离消融实验

## 目的

通过系统性 A/B 消融实验，**逐个隔离** QwenASR 流式参数（`chunk_sec`, `rollback`, `unfixed_chunks`, `max_new_tokens`），精确定位导致长音频 CER 严重退化的根因参数。

## 背景

Swift 产品端为降低首字延迟，将流式参数从 Rust 默认值调整为更激进的配置：

| 参数 | Rust 默认 | Swift 设置 | 调整目的 |
|------|:---------:|:----------:|----------|
| `chunk_sec` | 2.0s | 1.5s | 缩小窗口，降低首字延迟 |
| `rollback` | 5 | 3 | 减少回退，降低延迟 |
| `unfixed_chunks` | 2 | 1 | 减少冷启动期 |
| `max_new_tokens` | 32 | 32 | 保持默认 |

调整后，长音频（>9s）的流式 CER 从 ~0.01 飙升至 ~0.70。需要定位是哪个参数（或哪些参数的组合）导致了退化。

## 验证方法

### 测试矩阵

14 种参数配置，分四组：

**第一组：单参数隔离**（每次仅改一个参数，其余保持默认）
| # | 配置 | chunk | rb | uf | mnt |
|---|------|:-----:|:--:|:--:|:---:|
| A | 默认参数（基线） | 2.0 | 5 | 2 | 32 |
| B | Swift 全部自定义 | 1.5 | 3 | 1 | 32 |
| C | 仅改 chunk_sec=1.5 | 1.5 | 5 | 2 | 32 |
| D | 仅改 rollback=3 | 2.0 | 3 | 2 | 32 |
| E | 仅改 unfixed_chunks=1 | 2.0 | 5 | 1 | 32 |

**第二组：双参数组合**
| # | 配置 | chunk | rb | uf | mnt |
|---|------|:-----:|:--:|:--:|:---:|
| F | chunk=1.5 + rollback=3 | 1.5 | 3 | 2 | 32 |
| G | chunk=1.5 + unfixed=1 | 1.5 | 5 | 1 | 32 |
| H | rollback=3 + unfixed=1 | 2.0 | 3 | 1 | 32 |

**第三组：chunk_sec 梯度**
| # | 配置 | chunk |
|---|------|:-----:|
| I | chunk_sec=1.7 | 1.7 |
| J | chunk_sec=1.8 | 1.8 |
| K | chunk_sec=1.9 | 1.9 |

**第四组：chunk=1.5 + 不同 rollback**
| # | 配置 | rb |
|---|------|:--:|
| L | chunk=1.5 + rollback=4 | 4 |
| M | chunk=1.5 + rollback=6 | 6 |
| N | chunk=1.5 + rollback=8 | 8 |

### 测试数据

3 条长音频（长音频是截断问题最敏感的场景）：

| ID | 时长 | 说明 |
|----|:----:|------|
| zh_long_01 | ~9s | 中文长句 |
| long_30s_01 | ~43s | 多句段落 |
| long_60s_01 | ~75s | 12 句完整段落 |

## 结论与发现

### 核心结论：chunk_sec=1.5 是唯一导致退化的参数

#### 单参数隔离结果

| 配置 | chunk | rb | uf | zh_long | long_30s | long_60s | 平均 CER |
|------|:-----:|:--:|:--:|:-------:|:--------:|:--------:|:--------:|
| A. 默认参数 | 2.0 | 5 | 2 | 0.000 | 0.010 | 0.009 | **0.006** |
| C. 仅改 chunk_sec=1.5 | **1.5** | 5 | 2 | **0.537** | **0.665** | **0.912** | **0.705** |
| D. 仅改 rollback=3 | 2.0 | **3** | 2 | 0.000 | 0.015 | 0.012 | 0.009 |
| E. 仅改 unfixed_chunks=1 | 2.0 | 5 | **1** | 0.000 | 0.010 | 0.009 | 0.006 |

**只有 chunk_sec=1.5 导致 CER 从 0.006 跳变到 0.705**。rollback 和 unfixed_chunks 的调整对 CER 无有意义影响。

#### 双参数组合验证

| 配置 | 平均 CER | 退化？ |
|------|:--------:|:------:|
| F. chunk=1.5 + rb=3 | 0.702 | 是（跟随 chunk） |
| G. chunk=1.5 + uf=1 | 0.705 | 是（跟随 chunk） |
| H. rb=3 + uf=1 | **0.009** | **否** |

只要 `chunk_sec` 保持 2.0，其他参数任意调整均不影响长音频质量。

#### chunk_sec 梯度测试

| chunk_sec | 平均 CER |
|:---------:|:--------:|
| 2.0 | 0.006 |
| 1.9 | 0.103 |
| 1.8 | 0.067 |
| 1.7 | 0.135 |
| 1.5 | 0.705 |

CER 退化从 `chunk_sec < 2.0` 时开始出现，随 chunk_sec 减小急剧恶化。注意 CER 呈非单调关系（1.8 好于 1.9 和 1.7），与 chunk 边界和 encoder window 对齐有关。

#### 增大 rollback 无法挽救

| rollback | long_60s CER |
|:--------:|:------------:|
| 3 | 0.903 |
| 4 | 0.909 |
| 5 | 0.912 |
| 6 | 0.918 |
| 8 | 0.924 |

在 chunk_sec=1.5 下增大 rollback 完全无效，甚至略微恶化。

### 根因分析

**encoder window 对齐问题**：

```
encoder window = 800 × 160 = 128,000 samples = 8.0s
chunk_samples(2.0s) = 32,000 → 128,000 / 32,000 = 4.0（整除）
chunk_samples(1.5s) = 24,000 → 128,000 / 24,000 = 5.33（不整除）
```

chunk_sec=2.0 刚好整除 encoder window，每 4 个 chunk 精确填满一个 window。chunk_sec=1.5 无法整除，导致 partial encoder 反复编码不完整的 window 尾部，encoder 上下文碎片化，token 质量下降。

两种退化模式：
1. **截断**：decoder 在 re-anchor 后因 encoder context 不足迅速产生 EOS
2. **重复后截断**：partial encoder 碎片化导致 decoder 重复输出

### 推荐配置

```
chunk_sec=2.0, rollback=3, unfixed_chunks=1, max_new_tokens=32
```

保持默认 chunk_sec 保证质量，通过 rollback（5→3）和 unfixed_chunks（2→1）获得 ~2.3s 延迟收益，CER 仅从 0.006 微升至 0.009。

## 依赖

```bash
uv run scripts/param_isolation_test.py
```

需要：
- `~/Github/typeless/Frameworks/qwen-asr/lib/libqwen_asr.dylib`
- Qwen3-ASR-0.6B 模型（HuggingFace cache）
- 测试音频集（`typeless/Tests/fixtures/`）
