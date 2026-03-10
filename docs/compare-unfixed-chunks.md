# compare_unfixed_chunks.py — unfixed_chunks 参数对流式 ASR 质量的影响

## 目的

对比 `unfixed_chunks` 参数在不同取值下对 QwenASR 流式识别质量（CER）和性能（RTF）的影响。`unfixed_chunks` 控制流式识别冷启动期的长度——在前 N 个 chunk 内，decoder 输出不确认（不 emit），等待后续 chunk 纠错后再确认。

## 验证方法

### 测试配置

通过 Rust dylib (`libqwen_asr.dylib`) C API 调用 `qwen_asr_stream_push()` 进行真正的流式推理（与 Swift FFI 同路径）。

| 配置 | unfixed_chunks | 说明 |
|------|:--------------:|------|
| A（默认） | 2 | 前 2 个 chunk 输出不确认，等待后续纠错 |
| B（对照） | 0 | 完全不使用冷启动保护，从第 1 个 chunk 起就确认输出 |

其他参数保持 Rust 默认值（`chunk_sec=2.0`, `rollback=5`, `max_new_tokens=32`）。

### 测试数据

67 条测试音频（来自 `corpus.json` + `real_manifest.json`），跳过 `silence`、`silence_short`、`hallucination` 类别。每条以 2s chunk 推送，最后追加 0.1s silence + finalize。

### 指标

- **CER**（Character Error Rate）：归一化编辑距离 / 参考文本长度
- **RTF**（Real-Time Factor）：处理耗时 / 音频时长
- **逐条差异对比**：列出两种配置 CER 有差异的条目

## 结论与发现

### 1. unfixed_chunks 对长音频 CER 无有意义的影响

在参数隔离测试（`param_isolation_test.py`）中已证实：

| 配置 | chunk_sec | rollback | unfixed_chunks | 平均 CER |
|------|:---------:|:--------:|:--------------:|:--------:|
| 默认参数 | 2.0 | 5 | 2 | 0.006 |
| 仅改 unfixed_chunks=1 | 2.0 | 5 | 1 | 0.006 |
| rb=3 + uf=1 | 2.0 | 3 | 1 | 0.009 |

将 `unfixed_chunks` 从 2 降至 1 甚至 0，对 CER 几乎无影响。长音频退化的根因是 `chunk_sec`（详见 `param_isolation_test.py` 文档）。

### 2. unfixed_chunks=1 可降低冷启动延迟

`unfixed_chunks=2` 意味着前 2 个 chunk（约 4s）的识别结果不确认，用户需等待 ~4s 才看到第一个稳定输出。降低为 1 可将等待缩短至 ~2s，对实时交互体验有明显改善。

### 3. 实际采用的参数

基于本脚本和 `param_isolation_test.py` 的综合结论，Swift 产品端推荐配置为：

```
chunk_sec=2.0, rollback=3, unfixed_chunks=1, max_new_tokens=32
```

保持 `chunk_sec=2.0` 保证长音频质量，`rollback=3` 和 `unfixed_chunks=1` 降低延迟，CER 不退化。

## 依赖

```bash
uv run scripts/compare_unfixed_chunks.py
```

需要：
- `target/release/libqwen_asr.dylib`（Rust 编译产物）
- Qwen3-ASR-0.6B 模型（HuggingFace cache）
- 测试音频集（`typeless/Tests/fixtures/`）
