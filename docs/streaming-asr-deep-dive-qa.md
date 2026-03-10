# Qwen3-ASR 边听边转录（Streaming ASR）深度解析 — Q&A 篇

> 本文以问答形式，从零开始逐步推导 Qwen3-ASR 的实时流式转录方案，所有内容与本 repo 源码一一对应。

---

## 第一部分：音频前处理 — PCM → Mel 频谱图

### Q1: 原始音频是什么格式？

一维的 `f32` 数组，采样率 16kHz（`config.rs:3`）。2 秒音频就是 `f32[32000]`。

### Q2: Mel 频谱图是什么？

把一维的时域波形转成二维的时频表示。纵轴是频率（128 个 Mel 频带），横轴是时间（每帧 10ms）。

关键常量（`config.rs:3-5`）：
- `SAMPLE_RATE = 16000`
- `MEL_BINS = 128`
- `HOP_LENGTH = 160`（每帧跳 160 个采样点，160 / 16000 = 10ms）

### Q3: 2 秒音频产出的 Mel 频谱图是什么形状？

先做 padding（两端各补 `N_FFT/2 = 200` 个零），然后按 `audio.rs:293` 的公式：

```
padded_len = 32000 + 400 = 32400
n_frames = (32400 - 400) / 160 + 1 = 201
```

输出形状：**`[128, 201]`**，即 128 个频带 × 201 帧。

### Q4: 总结一下这一步做了什么？

```
PCM  f32[32000]       (一维，2 秒原始波形)
        ↓
Mel  f32[128 × 201]   (二维，128 频带 × ~200 帧)
```

---

## 第二部分：Encoder — Mel 频谱图 → encoder tokens

### Q5: Encoder 的整体结构是什么？

```
Mel [128, ~200]
  ↓ 三层 Conv2D stem（下采样）
  ↓ Reshape + 线性投影
  ↓ Transformer × 24 层（窗口注意力）
  ↓ proj1 + proj2（投影到 decoder 维度）
encoder tokens [~26, 1024]
```

见 `encoder.rs:243`，签名：`fn forward(&self, ..., mel, mel_frames) -> (Vec<f32>, usize)`。

### Q6: 三层 Conv2D 分别做了什么？

三层参数完全相同：`kernel=3×3, stride=2, padding=1`，通道数 `CONV_HIDDEN = 480`（`config.rs:23`）。

以 2s 音频（Mel `[128, 201]`）为例：

| 阶段 | 输入 | 输出 | 代码位置 |
|------|------|------|---------|
| Conv1 | [1, 128, 201] | [480, 64, 101] | `encoder.rs:291-299` |
| Conv2 | [480, 64, 101] | [480, 32, 51] | `encoder.rs:301-309` |
| Conv3 | [480, 32, 51] | [480, 16, 26] | `encoder.rs:311-320` |

每层之后都有 GELU 激活。

关键压缩效果：
- **频率轴**：128 → 64 → 32 → **16**（每层减半）
- **时间轴**：201 → 101 → 51 → **26**（每层约减半，约 **8 倍压缩**）

### Q7: Conv2D 之后的 Reshape + 投影是怎么做的？

Conv3 输出 `[480, 16, 26]`，代码在 `encoder.rs:322-338`：

1. **Reshape**：把通道维和频率维合并 → `[26, 480×16] = [26, 7680]`
2. **线性投影**（`conv_out_weight`，无 bias）：`[26, 7680] → [26, 1024]`（其中 `d_model = 1024`，见 `config.rs:126`）
3. **加正弦位置编码**（sinusoidal PE）：每个 chunk 独立从 0 开始编号（`encoder.rs:341-343`）

### Q8: Transformer 层做了什么？

24 层标准 pre-norm Transformer encoder（`encoder.rs:368-398`），不改变 token 数量。

每层结构：
```
LayerNorm → Q,K,V = Linear(x) → WindowedAttention → Linear → 残差
LayerNorm → FFN(GELU) → 残差
```

关键参数（0.6B 模型）：`d_model=1024`, `heads=16`, `head_dim=64`, `ffn_dim=4096`。

**窗口注意力**：不是全局 self-attention，而是分窗口计算，窗口大小由 `enc_n_window_infer=800` 帧对应的 token 数决定。

### Q9: 最终投影做了什么？

```
LayerNorm → proj1(Linear + GELU) → proj2(Linear)
[26, 1024] → [26, 1024] → [26, 1024]
```

`enc_output_dim` 等于 `dec_hidden`（`config.rs:141`），0.6B 模型都是 1024。

### Q10: 总结 Encoder 的完整压缩链路？

```
PCM  f32[32000]           (2s 音频)
  ↓  mel_spectrogram
Mel  f32[128 × 201]
  ↓  Conv1 (stride=2)     → [480, 64, 101]
  ↓  Conv2 (stride=2)     → [480, 32, 51]
  ↓  Conv3 (stride=2)     → [480, 16, 26]   ← 时间轴 8x 压缩
  ↓  reshape + linear     → [26, 1024]       ← 频率轴折叠进特征维
  ↓  Transformer ×24      → [26, 1024]       ← token 数不变，特征精炼
  ↓  proj1 + proj2        → [26, 1024]       ← 投影到 decoder 空间
```

**2 秒音频 → 26 个 encoder tokens**，每个 token 约代表 77ms 音频。

---

## 第三部分：Decoder — encoder tokens → 文字

### Q11: Decoder 是什么？

本质就是一个 **Qwen3 LLM**（大语言模型），与 ChatGPT 工作方式相同：给定输入 token 序列，自回归地逐 token 生成输出。

区别是：输入序列中不仅有文字 token，还有 encoder 输出的音频 token。

### Q12: Decoder 的输入序列长什么样？

代码中定义了三段常量（`transcribe.rs:12-14`）：

```rust
PREFIX_HEAD = [151644, 8948, 198]           // <|im_start|> system \n
PREFIX_TAIL = [151645, 198, 151644, 872, 198, 151669]  // <|im_end|> \n <|im_start|> user \n <|audio_start|>
SUFFIX_BASE = [151670, 151645, 198, 151644, 77091, 198] // <|audio_end|> <|im_end|> \n <|im_start|> assistant \n
```

完整的输入序列：

```
PREFIX_HEAD:   <|im_start|> system \n
PREFIX_TAIL:   <|im_end|> \n <|im_start|> user \n <|audio_start|>
               [enc_token_0] [enc_token_1] ... [enc_token_25]    ← encoder 输出
SUFFIX_BASE:   <|audio_end|> <|im_end|> \n <|im_start|> assistant \n
               <|asr_text|>
```

### Q13: `[AUDIO_START]` 和 `[AUDIO_END]` 之间是什么？

**全部是 encoder 的输出向量**（1024 维 float），不是 decoder 生成的 token。

系统中存在两种不同的"token"：
- **文字 token**：有 token ID，通过 embedding 查表（`tok_embed_to_f32`）得到 1024 维向量
- **音频 token**：没有 token ID，是 encoder 直接输出的 1024 维向量，直接 `copy_from_slice` 拼入 `input_embeds`

代码（`transcribe.rs:575-593`）：
```rust
// 文字 token → 查表
for &tok in PREFIX_HEAD {
    ctx.tok_embed_to_f32(&mut input_embeds[off..], tok, dim);
}
// 音频 token → 直接拷贝
for i in 0..enc_seq_len {
    input_embeds[...].copy_from_slice(&enc_output[i * dim..]);
}
```

### Q14: Decoder 有哪两种关键的工作模式？

**1. Prefill（批量处理）— `decoder_prefill`（`decoder.rs:412`）**

一次性处理整个输入序列，填充 KV Cache。不直接产出 token。

**2. Forward（单 token 生成）— `decoder_forward`（`decoder.rs:507`）**

每次只处理 1 个 token，利用 KV Cache 中缓存的历史 K/V，输出下一个 token ID（argmax）。

### Q15: 完整的离线转录（非流式）流程是什么？

```
Step 1 — Prefill:
  拼好输入序列 (prefix + encoder tokens + suffix + <|asr_text|>)
  调用 decoder_prefill → 填充 KV Cache

Step 2 — 自回归生成循环:
  token = decoder_forward(上一个 token 的 embedding)
  if token == <|im_end|>:  停止
  else: 记录 token，继续

Step 3 — 解码:
  token ID 序列 → tokenizer.decode → "你好世界"
```

### Q16: Decoder 每层 Transformer 的内部结构？

代码 `decoder.rs:539-586`，0.6B 模型参数：`hidden=1024, heads=16, kv_heads=8, head_dim=128, intermediate=3072, layers=28`。

```
Input x [1, 1024]
  ↓ RMSNorm
  ↓ Q = Linear(x) → [1, 2048]  (16 heads × 128 dim)
  ↓ K = Linear(x) → [1, 1024]  (8 KV heads × 128 dim)  ← GQA: Q/K head 数不同
  ↓ V = Linear(x) → [1, 1024]
  ↓ RMSNorm per head (Q_norm, K_norm)
  ↓ RoPE 旋转位置编码
  ↓ 存 K,V 到 KV Cache
  ↓ Causal Attention (attend 所有历史 K,V)
  ↓ O = Linear(attn_out) → [1, 1024]
  ↓ x = x + O                 ← 残差连接
  ↓ RMSNorm
  ↓ SwiGLU FFN: gate_up → silu(gate)×up → down
  ↓ x = x + ffn_out           ← 残差连接
Output x [1, 1024]
```

关键设计：
- **GQA**：16 个 Q head 共享 8 组 K/V，节省 KV Cache 内存
- **RoPE**：旋转位置编码，编码绝对位置信息
- **Causal Attention**：只能看到当前位置之前的 token
- **SwiGLU**：`silu(gate) × up` 门控激活

### Q17: KV Cache 是什么？为什么重要？

定义在 `decoder.rs:136-234`，结构：`k[n_layers × max_seq × kv_dim]`，`v` 同理。

作用：避免重复计算。没有 KV Cache，生成第 N 个 token 需要重新计算前面所有位置的 K、V。有了 KV Cache：

```
Prefill:        一次算好所有输入位置的 K,V，存进 cache
生成 token 1:   只算 1 个位置的 Q,K,V，K/V 追加到 cache
生成 token 2:   只算 1 个位置，attend 到 cache 中所有历史
...
```

每生成一个 token，`kv_cache.len` 加 1（`decoder.rs:588`）。

---

## 第四部分：Streaming — 核心设计

### Q18: 如果让你实现"边听边转"，最朴素的想法是什么？

> 每收到一小段音频（如 2 秒），把到目前为止所有音频拼起来，重新跑 Encoder + Decoder，得到最新文字。

这个想法正确，但**随着音频增长，计算量线性增长**，说了 1 分钟后每来 2 秒就要重新编码 1 分钟，延迟不可接受。

### Q19: Encoder 端如何解决重复编码问题？

**Encoder 天然支持分块处理**。三层 Conv2D 的位置编码（sinusoidal PE）每个 chunk 独立从 0 开始，不依赖全局位置。

代码中的做法（`transcribe.rs:488-553`）：

```
已完成的窗口 → 存入 enc_cache（编码一次，永不重算）
尾部不足一个窗口 → 每轮重新编码（partial encoding）
```

encoder 窗口大小 = `enc_window_frames(800) × HOP_LENGTH(160) = 128000 采样 = 8 秒`。

### Q20: Decoder 端面临什么挑战？

新 chunk 来了之后，encoder tokens 数量增多，输入序列变成：

```
Chunk 0: prefix(9) + enc(26)  + suffix(6) = 41
Chunk 1: prefix(9) + enc(52)  + suffix(6) = 67
Chunk 2: prefix(9) + enc(78)  + suffix(6) = 93
```

encoder tokens 插在序列中间，suffix 的位置变了。由于 RoPE 依赖绝对位置，上一轮 KV Cache 中 suffix 对应的 K/V **不再正确**，不能直接复用。

### Q21: 这个 repo 用了什么方案解决？

**全量重建 + LCP（Longest Common Prefix）复用**。

核心思路：每轮仍然构建完整的 `input_embeds`，但逐行比较本轮和上一轮的向量，找到从头开始的最长公共前缀，**跳过这部分的 prefill**。

代码 `transcribe.rs:620-654`，三步：

**Step 1 — 逐行比较找 LCP：**
```rust
while reused_prefill < cmp_len {
    let a = &prev_prefill_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
    let b = &input_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
    if a != b { break; }
    reused_prefill += 1;
}
```

**Step 2 — 截断 KV Cache 到 LCP 断点：**
```rust
ctx.kv_cache.len = reused_prefill;
```

**Step 3 — 只 prefill 变化的部分：**
```rust
let delta_prefill = prefill_len - reused_prefill;
ctx.decoder_prefill(&input_embeds[reused_prefill * dim..], delta_prefill);
```

### Q22: LCP Reuse 到底能复用多少？

举例，Chunk 1 → Chunk 2：

```
Chunk 1: [prefix(9)] [enc_win0(26)] [enc_win1(26)] [suffix(6)] [text(5)]
Chunk 2: [prefix(9)] [enc_win0(26)] [enc_win1(26)] [enc_win2(26)] [suffix(6)] [text(8)]
```

从头比较：
- prefix 9 个 token：**完全相同**（固定的特殊 token 查表结果）
- enc_win0 + enc_win1 共 52 个：**完全相同**（encoder 缓存了旧窗口，向量一模一样）
- 位置 61 开始不同（Chunk1 是 suffix，Chunk2 是 enc_win2）

所以 `reused_prefill = 61`，只需 prefill 后面 `26 + 6 + 8 = 40` 个 token，而非全部 101 个。

**prefix + 已缓存 encoder tokens 的 KV Cache 全部直接复用。**

### Q23: Rollback 机制是什么？

因为音频还没说完，decoder 生成的最后几个 token 可能不准确。每轮只将前面部分确认为"稳定输出"，尾部 `rollback` 个 token 丢弃重新生成。

代码 `transcribe.rs:558-563`：
```rust
let n_prefix_tokens = if chunk_idx >= unfixed_chunks && !raw_tokens.is_empty() {
    (raw_tokens.len() - rollback).max(0)
} else {
    0
};
```

示例（`rollback=2`）：
```
Chunk 1 生成: [你, 好, 世, 界, 欢]
  → 确认输出: [你, 好, 世]       ← 通过 callback 发给用户
  → 丢弃:     [界, 欢]           ← 下轮作为前缀重新生成

Chunk 2 输入: prefix + enc_tokens + suffix + [你, 好, 世]
Chunk 2 生成: [界, 欢, 迎, 你]
  → 确认输出: [界, 欢]
  → 丢弃:     [迎, 你]

最终累计输出: "你好世界欢"（逐步追加）
```

---

## 第五部分：Streaming 安全机制

### Q24: 什么是退化检测（Degeneracy Detection）？

Decoder 可能陷入重复生成的死循环（如 "的的的的..."）。代码通过两种方式检测（`transcribe.rs:678-688`）：

1. **Stale 检测**：连续 `STREAM_STALE_CHUNKS=4` 个 chunk 的输出完全相同
2. **循环块检测**（`stream_tail_repeat_blocks`）：检查 token 尾部是否存在长度 ≤ `STREAM_DEGEN_MAX_PERIOD=6` 的重复块，重复 ≥ `STREAM_DEGEN_MIN_REPEATS=4` 次

触发后执行 reset：清空 raw_tokens、丢弃旧的 encoder 窗口（只保留最近 2 个）、清空 prefill 缓存。

### Q25: 什么是 Re-anchor 机制？

当累计的 encoder tokens 太多（≥ `STREAM_REANCHOR_ENC_SEQ_THRESHOLD=200`，约 15 秒音频）或达到固定间隔（`STREAM_RESET_INTERVAL_CHUNKS=45`），触发 re-anchor（`transcribe.rs:725-757`）。

Re-anchor 做的事：
- 保留最近 1 个 encoder 窗口，丢弃更老的
- 清空 KV Cache 和 prefill 缓存
- 携带最近 `STREAM_RESET_CARRY_TOKENS=24` 个已确认 token 作为上下文

目的是防止 decoder 输入序列无限增长导致性能下降或退化。

### Q26: Re-anchor 丢弃旧音频不会丢信息吗？

不会，因为旧音频对应的文字已经通过 callback 输出给用户了（`stable_text_tokens` 中已确认的部分）。Re-anchor 只是让 decoder 丢掉过远的音频上下文，集中注意力在最近的音频上。携带的 24 个 carry tokens 保证了文字的连贯性。

---

## 第六部分：总结

### Q27: 一图总结完整 Streaming 流程？

```
每来一个 2s chunk:

1. Encoder 端
   ├─ 已完成的窗口 → 从 enc_cache 取（不重算）
   └─ 尾部 partial → 重新编码
   拼接得到完整 enc_output [N, 1024]

2. 构建完整 input_embeds
   [prefix(9)] + [enc_output(N)] + [suffix(6)] + [confirmed_text(M)]

3. LCP Reuse
   ├─ 与上一轮 input_embeds 逐位比较 1024 维向量
   ├─ 找到第一个不一致的位置 → reused_prefill
   ├─ KV Cache 截断到 reused_prefill
   └─ 只 prefill 变化的尾部

4. 自回归生成（最多 max_new_tokens=32 个 token）

5. Rollback + 确认
   ├─ 前面的 tokens → 确认，通过 token_cb 输出
   └─ 最后 rollback 个 → 下轮重新生成

6. 安全机制
   ├─ 退化检测 → 发现重复循环就 reset
   └─ Re-anchor → encoder 序列过长就裁剪旧窗口
```

### Q28: 核心源文件对照表？

| 文件 | 职责 |
|------|------|
| `config.rs` | 常量定义（SAMPLE_RATE, MEL_BINS, HOP_LENGTH, token IDs）和模型配置 |
| `audio.rs` | WAV 解析、重采样、Mel 频谱图计算 |
| `encoder.rs` | Conv2D stem + Transformer encoder + 投影 |
| `decoder.rs` | Qwen3 LLM decoder（prefill + forward），KV Cache，RoPE |
| `transcribe.rs` | 转录编排（离线/分段/流式），LCP Reuse，rollback，退化检测，re-anchor |
| `tokenizer.rs` | BPE tokenizer，token ID ↔ 文字相互转换 |
| `c_api.rs` | C-FFI 接口（移动端/原生集成） |

### Q29: 关键设计参数速查？

| 参数 | 值 | 含义 |
|------|----|------|
| `SAMPLE_RATE` | 16000 | 音频采样率 |
| `MEL_BINS` | 128 | Mel 频带数 |
| `HOP_LENGTH` | 160 | 帧跳步（10ms/帧） |
| `CONV_HIDDEN` | 480 | Conv stem 通道数 |
| `enc_chunk_size` | 100 | Encoder 分块大小（帧） |
| `enc_n_window_infer` | 800 | Encoder 推理窗口大小（帧，≈8 秒） |
| `enc_d_model` | 1024 | Encoder 隐藏维度 |
| `enc_layers` | 24 | Encoder Transformer 层数 |
| `dec_hidden` | 1024 | Decoder 隐藏维度 |
| `dec_layers` | 28 | Decoder Transformer 层数 |
| `dec_heads` | 16 | Decoder Q 注意力头数 |
| `dec_kv_heads` | 8 | Decoder K/V 注意力头数（GQA） |
| `dec_head_dim` | 128 | 每个注意力头的维度 |
| `stream_chunk_sec` | 2.0 | Streaming 音频分块大小（秒） |
| `STREAM_REANCHOR_ENC_SEQ_THRESHOLD` | 200 | Re-anchor 触发阈值（≈15 秒） |
| `STREAM_DEGEN_MIN_REPEATS` | 4 | 退化检测：最少重复次数 |
| `STREAM_STALE_CHUNKS` | 4 | 退化检测：连续输出不变的 chunk 数 |
| `STREAM_RESET_CARRY_TOKENS` | 24 | Reset 时携带的上下文 token 数 |
