# Metal GPU 加速优化记录

*测试环境：Apple Silicon (M-series), macOS, Qwen3-ASR-0.6B (BF16 safetensors)*
*测试音频：audio.wav, 28.2s, 英文*
*每项取 3 次运行中较稳定的值（排除首次 shader 编译）*

---

## 基线：CPU-only (Accelerate vDSP + NEON + 14 线程)

| 模式 | Encoding | Decoding | Total | Realtime |
|------|:--------:|:--------:|:-----:|:--------:|
| Offline | 443ms | 1254ms | 1697ms | 16.6x |
| Streaming | 1312ms | 3761ms | 5074ms | 5.6x |

---

## Phase 0 + Phase 1 v1：Encoder GPU (candle Metal) — QKV GPU + Attention CPU 往返

将 encoder transformer 层的 Linear (QKV, output proj, FFN) 迁移到 Metal GPU。
**但** windowed bidirectional attention 仍在 CPU 执行，每层需要 3 次 GPU→CPU 拷贝 (Q/K/V) + 1 次 CPU→GPU 拷贝 (attention output)。

| 模式 | Encoding | Decoding | Total | Realtime | vs CPU |
|------|:--------:|:--------:|:-----:|:--------:|:------:|
| Offline | 462ms | 1189ms | 1651ms | 17.1x | -0.5x |
| Streaming | 1606ms | 3558ms | 5165ms | 5.5x | **-22% 更慢** |

**结论**：GPU↔CPU 数据往返开销完全抵消了 GPU matmul 的收益。每层 4 次跨设备拷贝是瓶颈。

---

## Phase 1 v2：全 GPU Encoder (含 GPU attention)

实现 `windowed_attention_gpu()`：将 windowed bidirectional attention 完全在 GPU 端用 candle tensor 操作实现。
消除了每层 4 次 GPU↔CPU 拷贝，整个 encoder transformer 现在只有输入和输出各 1 次跨设备传输。

关键修复：candle Metal 不支持非连续 tensor 的 matmul，需要在 transpose 后调用 `.contiguous()`。
同时 layer_norm 和 softmax 放在 CPU 执行（避免 Metal broadcast 兼容性问题）。

| 模式 | Encoding | Decoding | Total | Realtime | vs CPU |
|------|:--------:|:--------:|:-----:|:--------:|:------:|
| Offline | 498ms | 1126ms | 1624ms | 17.3x | **+5% 更快** |
| Streaming | 1717ms | 3538ms | 5256ms | 5.4x | **-4% 更慢** |

**分析**：
- **Offline encoding**：498ms vs CPU 443ms (+12%)。GPU 开销略大于 matmul 加速收益，因为 layer_norm/softmax 仍 CPU→GPU 拷贝
- **Decoder 改善**：1126ms vs CPU 1254ms (-10%)。bidirectional attention 不再竞争 CPU 线程，decoder 的 `bf16_matvec` 性能提升
- **Streaming 更慢**：每 chunk ~25 tokens，batch 太小，GPU kernel launch 开销主导

**根因**：当前 encoder GPU 路径仍有大量 CPU↔GPU 拷贝：
1. Conv2D stem 在 CPU — 输出上传到 GPU ✓
2. 每层 layer_norm 在 CPU — 2 次下载+上传（进出 layer_norm_on_cpu）
3. softmax 在 CPU — attention scores 下载 + 结果上传
4. 最终输出下载回 CPU ✓

**每层 transformer 实际有 4 次 GPU↔CPU 数据搬运**（2 次 layer_norm + 1 次 softmax + 通过 matmul 结果）。

---

## 下一步优化方向

### 消除 layer_norm / softmax 的 CPU fallback
candle Metal 的 broadcast 操作有兼容性问题。可能的解决方案：
1. 用 candle `Tensor` ops 重新实现 layer_norm，避免 `broadcast_sub/mul`（用显式 reshape + repeat 代替 broadcast）
2. 升级 candle 版本（如果更新版本修复了 Metal broadcast）
3. 自定义 Metal shader 实现 layer_norm / softmax

### Decoder GPU 化
Decoder 是主要耗时（offline 1254ms，streaming 3761ms）。但单 token decode 的 GPU 化挑战更大：
- 每步只处理 1 token，GPU 并行效率极低
- KV cache 需要常驻 GPU
- `bf16_matvec`（向量×矩阵）不是 GPU 的强项

### 最小 token 数阈值
当 token 数 < 某个阈值时 fallback 到 CPU encoder，避免 GPU 在小 batch 上的开销。

---

## 硬件利用率分析

| 组件 | CPU (Accelerate) | GPU (candle Metal) |
|------|:-----------------:|:------------------:|
| Linear (矩阵乘) | cblas_sgemm (14线程) | Metal matmul shader |
| Layer Norm | SIMD (单线程) | **CPU fallback** |
| GELU | SIMD | gelu_erf Metal kernel |
| Softmax | SIMD | **CPU fallback** |
| Attention | 多线程 dot+scale | batched matmul on GPU |
| Conv2D | im2col + sgemm | **CPU (conv stem)** |

对于当前的 0.6B encoder (18 layers, d_model=896, 14 heads)，Linear 层是主要耗时 (~70%)。
但 CPU Accelerate cblas_sgemm 已经非常高效（利用了 AMX 协处理器），GPU matmul 很难大幅超越。

---

## Phase 2：Decoder GPU Prefill (BF16 Linear on Metal)

### 策略
Decoder prefill 处理 ~381 tokens 的 batch linear 运算（QKV, O, gate_up, down），是理想的 GPU 加速目标。
- 所有 6 个 Linear 层（QKV, O, gate_up, down）在 GPU 上执行
- RMSNorm, per-head norm, RoPE, causal attention, SwiGLU 保持 CPU
- KV cache 保持 CPU（与单 token decode 共享）

### 问题：过多的 GPU↔CPU 数据拷贝
每层 transformer 有 **6 次 GPU↔CPU round-trip**：
1. x_norm 上传 → QKV matmul → Q/K/V 下载（1 上传 + 3 下载）
2. attn_out 上传 → O proj → 下载（1 round-trip）
3. x_norm2 上传 → gate_up → 下载（1 round-trip）
4. gate 上传 → down proj → 下载（1 round-trip）

28 layers × 6 = **168 次 GPU↔CPU 拷贝**

### 结果

| 模式 | Encoding | Decoding | Total | Realtime | vs CPU |
|------|:--------:|:--------:|:-----:|:--------:|:------:|
| Offline | 479ms | 1136ms | 1615ms | 17.4x | **+5% 更快** |
| Streaming | 1712ms | 3388ms | 5101ms | 5.5x | **≈持平** |

**分析**：
- **Offline decoding 改善**：1136ms vs CPU 1254ms (-9.4%)，GPU matmul 仍有收益
- **但改善有限**：168 次 GPU↔CPU 拷贝的开销几乎抵消了 GPU matmul 的加速
- **Streaming 无改善**：encoder GPU 的开销 (+400ms) 与 decoder prefill 的加速 (-373ms) 互相抵消
- **对比 Phase 1 v2**：Offline total 1615ms vs 1624ms（差异在误差范围内），说明 decoder prefill GPU 的净收益很小

### 关键瓶颈
CPU Accelerate cblas_sgemm 利用 AMX 协处理器，对于 `[381, 1024] × [1024, 2048]` 级别的矩阵运算已经非常高效。
candle Metal matmul 通过通用 Metal shader 执行，无法利用 AMX。
加上 GPU↔CPU 拷贝开销，GPU 路径难以超越 CPU baseline。

---

## 结论与反思

经过 Phase 0 → Phase 2 的逐步 GPU 迁移实验，得到关键发现：

### Apple Silicon 的特殊性
1. **统一内存架构的陷阱**：虽然 CPU/GPU 共享物理内存，但 candle 的 GPU tensor 仍需要显式 copy（Tensor::to_device），产生实际的数据搬运和 GPU command buffer submission 开销
2. **AMX 协处理器**：CPU 端 Accelerate cblas_sgemm 已经利用 AMX（矩阵运算加速器），性能接近 GPU matmul，使得纯通过"CPU→GPU"切换获益有限
3. **小模型劣势**：0.6B 参数的矩阵尺寸（~1024×2048）对 GPU 而言 batch 不够大，GPU 的大规模并行优势未能充分发挥

### candle Metal 的限制
1. 不支持 broadcast_sub/mul/div — 导致 layer_norm/softmax 必须 CPU fallback
2. 非连续 tensor 的 matmul 会出错 — 需要额外 .contiguous() 拷贝
3. 无 fused kernel — 每个 op 单独 dispatch，无法做 QKV+norm+RoPE 融合
4. 无 Flash Attention — 标准 matmul-based attention，O(seq²) 内存

### MLX 快 3x 的真正原因
MLX 的优势不仅在于"用了 GPU"，更在于：
- **Zero-copy unified memory**：MLX array 真正利用了统一内存，无需 to_device
- **Lazy evaluation + 图优化**：MLX 自动融合 kernel，减少 dispatch 次数
- **专用 Metal shader**：MatMul、LayerNorm、RoPE 等有针对 Apple GPU 优化的 MSL 实现
- **Metal FFI 直接调用**：无 candle 抽象层开销

---

## Phase 3：CustomOp 直调 Metal Kernel（突破 candle 高层 API 限制）

### 背景

Phase 0-2 的核心教训：candle 高层 API 不支持 Metal 端的 LayerNorm/Softmax/RMSNorm/SwiGLU，导致每层 transformer 需要大量 CPU↔GPU round-trip。

### 关键发现：candle-metal-kernels 暗藏可用的原生 Metal shader

深入 candle 源码后发现，`candle-metal-kernels` crate 自带高效的 Metal shader 实现：
- `call_last_softmax()` — softmax 沿最后一维
- `call_layer_norm()` — LayerNorm (带 scale + bias)
- `call_rms_norm()` — RMSNorm (带 scale)

但 candle 的高层 `Tensor` API **没有暴露这些方法给 Metal backend**。解决方案：通过 `CustomOp` trait 直接调用底层 Metal kernel。

### 实现：metal_ops.rs

创建 `SoftmaxLastOp`、`LayerNormOp`、`RmsNormOp` 三个 CustomOp，绕过 candle Tensor API 直接 dispatch Metal shader：

```rust
// 示例：RmsNormOp
impl CustomOp2 for RmsNormOp {
    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout) -> Result<(MetalStorage, Shape)> {
        candle_metal_kernels::call_rms_norm(
            device.device(), &encoder, device.kernels(), kernel_name,
            length, elements_to_sum, self.eps,
            s1.buffer(), l1.start_offset() * bsz,
            s2.buffer(), l2.start_offset() * bsz,
            &output,
        )
    }
}
```

### 踩坑 #1：`MetalStorage::new()` 接受 `Arc<Buffer>` 而非 `Buffer`

`device.new_buffer()` 返回 `Result<Arc<Buffer>>`，而 `MetalStorage::new()` 签名是 `(Arc<Buffer>, MetalDevice, usize, DType)`。一开始以为 `buffer()` 返回 `Buffer` 可以直接 clone，实际 `buffer()` 返回 `&Buffer`（从 `Arc<Buffer>` deref）。

**解决**：直接把 `device.new_buffer()?` 的返回值传给 `MetalStorage::new()`，无需额外 clone。

### 踩坑 #2：CustomOp 的 `cpu_fwd` 必须实现

即使我们只用 GPU path，CustomOp trait 的 `cpu_fwd` 方法是 **required**（非 optional），否则编译失败。需要提供一个完整的 CPU fallback 实现。

### 成果

- Encoder: 用 `LayerNormOp` 替代 `layer_norm_on_cpu()`，用 `SoftmaxLastOp` 替代 `softmax_last()`
- Decoder prefill: 用 `RmsNormOp` + 额外的 `per_head_rms_norm_gpu()` + `swiglu_gpu()`
- 消除了 encoder 和 decoder 中所有 norm/softmax 的 CPU↔GPU round-trip

### Benchmark（Phase 3，67 条音频测试集）

| Pipeline | 平均 CER | RTF | 总推理时长 |
|----------|:-------:|:---:|:---------:|
| Offline (Phase 3 GPU) | 0.0571 | 0.163x | 68.6s |
| Streaming (Phase 3 GPU) | 0.0553 | 0.255x | 107.5s |
| Offline (CPU baseline) | 0.0571 | 0.090x | 37.9s |
| Streaming (CPU baseline) | 0.0553 | 0.178x | 75.0s |

**结论**：GPU 路径仍然比 CPU 慢 ~80%！CER 完全一致证明正确性没问题，但性能远不如预期。

### 根因分析

Decoder prefill 每层 transformer 仍有 **~8 次 GPU↔CPU round-trip**：
1. Download Q (normed, for RoPE) — CPU 做 RoPE
2. Download K (normed, for RoPE) — CPU 做 RoPE
3. Download V — 直接写 CPU KV cache
4. Write K to CPU KV cache
5. Write V to CPU KV cache
6. Causal attention on CPU (读 KV cache)
7. Upload attention output → O projection
8. Download FFN output → residual on CPU

28 layers × 8 = **~224 次 GPU↔CPU round-trip**，完全吞噬了 GPU 加速的收益。

---

## Phase 4：全 GPU Forward Pass（RoPE + SDPA + KV Cache on Metal）

### 目标

彻底消除 decoder prefill 中的层内 round-trip，将 RoPE、注意力（SDPA）、KV cache 全部迁移到 GPU。

### 关键 API 发现

#### 踩坑 #3：`call_sdpa_full` 实际支持 GQA（尽管注释说不支持）

candle-metal-kernels 0.9.2 中 `call_sdpa_full` 的源码注释写着 "GQA not supported"，但阅读实际代码发现它确实计算了 `gqa_factor = q_heads / kv_heads` 并正确处理。错误的注释差点让我们放弃使用这个 kernel。

Qwen3-ASR-0.6B: `n_heads=16, n_kv_heads=8, head_dim=128` → GQA factor=2，完全支持。

#### 踩坑 #4：`SdpaDType` 的导入路径

一开始以为 `SdpaDType` 不从 crate root re-export（因为 Explore agent 报告说不在 lib.rs 中）。实际上 lib.rs 第 12 行有 `pub use kernels::{ ... sdpa::* ... }`，所以 `candle_metal_kernels::SdpaDType::F32` 可以直接用。

**教训**：对 agent 的搜索结果需要交叉验证，特别是涉及 glob re-export (`::*`) 的情况。

#### 踩坑 #5：`Tensor::from_storage_no_op` 不存在

计划中假设可以用 `Tensor::from_storage_no_op(storage, shape)` 从 raw `Storage` 构造 Tensor。实际上 candle 只有：

```rust
pub fn from_storage<S: Into<Shape>>(
    storage: Storage,
    shape: S,
    op: BackpropOp,
    is_variable: bool,
) -> Tensor
```

**解决**：使用 `Tensor::from_storage(storage, shape, BackpropOp::none(), false)`。

#### 踩坑 #6：`[usize; 4]` 不实现 `Into<Shape>`

```rust
let o_shape = [1, n_heads, seq_len, head_dim]; // 编译错误
```

candle 的 `Shape::from` 只实现了 `&[usize]` 和 `Vec<usize>`，**不支持固定大小数组**。

**解决**：改用 `vec![1, n_heads, seq_len, head_dim]`。

#### 踩坑 #7：SDPA 函数中 `storage_and_layout()` 的生命周期问题

最初尝试通过 helper 函数提取 Metal buffer：

```rust
fn extract_metal_info(t: &Tensor) -> (Buffer, usize, Vec<usize>, Vec<usize>, MetalDevice) {
    let (guard, layout) = t.storage_and_layout(); // RwLockReadGuard
    match &*guard {
        Storage::Metal(ms) => {
            let buf = ms.buffer().clone(); // &Buffer 不能 clone 成 owned Buffer!
            ...
        }
    }
}
```

`buffer()` 返回 `&Buffer`（从 `Arc<Buffer>` deref），`Buffer` 不实现 `Clone`。即使能 clone，guard 释放后引用也无效。

**解决**：放弃 helper 函数，让 SDPA 函数在同一作用域中持有 Q/K/V 三个 guard，直接用 `buffer()` 引用传给 kernel：

```rust
pub fn sdpa_full_gpu(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (q_guard, q_layout) = q.storage_and_layout();
    let (k_guard, k_layout) = k.storage_and_layout();
    let (v_guard, v_layout) = v.storage_and_layout();
    // 三个 guard 同时活着，buffer() 引用有效
    let q_ms = match &*q_guard { Storage::Metal(ms) => ms, ... };
    candle_metal_kernels::call_sdpa_full(
        ..., q_ms.buffer(), ..., k_ms.buffer(), ..., v_ms.buffer(), ...
    )?;
    drop(q_guard); drop(k_guard); drop(v_guard);
    // 构造输出 Tensor
}
```

#### 踩坑 #8：`blit_command_encoder` 的 `copy_from_buffer` 参数是 `usize` 不是 `u64`

一开始把偏移量和长度都用 `as u64` 转换，因为想到 Metal API 通常用 64 位。结果 candle 的 wrapper `copy_from_buffer` 签名是全 `usize`。

```rust
// 错误：
let copy_bytes = (seq_len * hd * bsz) as u64; // ❌ 类型不匹配

// 正确：
let copy_bytes = seq_len * hd * bsz; // usize ✓
```

### 踩坑 #9（严重）：Metal `rope_thd` kernel 的 cos/sin 布局 ≠ CPU RopeCache 布局

这是最严重的 bug，导致 benchmark 全部 CER=1.000（完全错误的结果）。

**CPU RopeCache** 存储 cos/sin 为 `[position, head_dim]`，每行有重复：
```
cos[pos * head_dim + d] = cos_val  (d = 0..half)
cos[pos * head_dim + half + d] = cos_val  (重复，same values)
```

**Metal `rope_thd` kernel** 按 `[t, d/2]` 索引 cos/sin：
```metal
size_t i_cs = i_t * (d / 2) + i_d;  // Metal shader 源码
T c = cos[i_cs];
T s = sin[i_cs];
```

一开始直接把 CPU 的 `[t, head_dim]` 上传到 GPU，kernel 读 `cos[t * (d/2) + d_idx]` 时，在 CPU 的 `[t, d]` 布局中对应了错误的位置。

**Root cause 确认**：通过阅读 Metal shader 源码（`.metal` 文件第 1423-1452 行）才发现 indexing 公式。Rust wrapper 的参数名和注释都没有说明这个细节。

**解决**：`GpuRopeCache::ensure()` 中只提取每行的前 `head_dim/2`：

```rust
let half = hd / 2;
let mut cos_half = vec![0.0f32; new_cap * half];
for pos in 0..new_cap {
    cos_half[pos * half..(pos + 1) * half]
        .copy_from_slice(&cpu_rope.cos[pos * hd..pos * hd + half]);
}
self.cos = Tensor::from_slice(&cos_half, &[new_cap, half], &Device::Cpu)?.to_device(&self.device)?;
```

同时 `RopeOp` 的 `cpu_fwd` 也要适配 `[seq, d/2]` 布局（不能再调用 `apply_rope_neox` 了）。

### 踩坑 #10（严重）：GPU prefill 成功但后续 CPU decode 读到空的 KV cache

Phase 4 只把 prefill 迁移到 GPU，后续单 token decode 仍用 CPU。但 CPU decode 的 `decoder_forward()` 从 CPU `KvCache` 读取数据——而全 GPU prefill 的 K/V 数据只存在 GPU 的 `GpuKvCache` 中，CPU KV cache 是空的！

**现象**：CER=1.000 + 每个文件耗时 ~40 秒（首次 shader 编译）。prefill 输出 hidden state 是对的，但 decode 循环因为空 KV cache 产生垃圾结果。

**解决**：在 GPU prefill 成功后，添加 `gpu_kv.sync_to_cpu(cpu_kv)` 将 GPU KV cache 下载到 CPU：

```rust
pub fn sync_to_cpu(&self, cpu_kv: &mut KvCache) -> Result<()> {
    for layer in 0..self.n_layers {
        let k_view = self.k[layer].narrow(2, 0, total_seq)?;
        let k_data = k_view.contiguous()?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        // 转换 [n_kv_heads, total_seq, head_dim] → [total_seq, n_kv_heads * head_dim] CPU 布局
        for s in 0..total_seq {
            for h in 0..n_kv {
                let gpu_off = h * total_seq * hd + s * hd;
                let cpu_off = (layer * cpu_kv.max_seq + s) * kv_dim + h * hd;
                cpu_kv.k[cpu_off..cpu_off + hd].copy_from_slice(&k_data[gpu_off..gpu_off + hd]);
            }
        }
    }
}
```

注意 GPU 布局是 `[n_kv_heads, seq, head_dim]` 而 CPU 是 `[seq, n_kv_heads * head_dim]`，需要做维度转换。

### 踩坑 #11：Streaming 模式中 GPU KV cache 的 len 必须同步

Streaming 模式会复用之前 prefill 的 KV cache（`ctx.kv_cache.len = reused_prefill`），只对增量部分做 prefill。如果不同步 GPU KV cache 的 `len` 字段，GPU 会从错误的 position 开始写入。

**解决**：在 `transcribe.rs` 中所有 `ctx.kv_cache.len = ...` 的位置都加上 GPU 同步：

```rust
ctx.kv_cache.len = reused_prefill;
#[cfg(feature = "metal")]
if let Some(ref mut gpu_kv) = ctx.gpu_kv_cache {
    gpu_kv.len = reused_prefill;
}
```

共有 3 处需要同步（offline reset × 1, streaming reuse × 2）。

### Phase 4 最终 Benchmark

| Pipeline | 平均 CER | RTF | 总推理时长 |
|----------|:-------:|:---:|:---------:|
| **Offline (Phase 4 GPU)** | **0.0571** | **0.079x** | **33.3s** |
| **Streaming (Phase 4 GPU)** | **0.0562** | **0.156x** | **65.6s** |
| Offline (Phase 3 GPU) | 0.0571 | 0.163x | 68.6s |
| Streaming (Phase 3 GPU) | 0.0553 | 0.255x | 107.5s |
| Offline (CPU baseline) | 0.0571 | 0.090x | 37.9s |
| Streaming (CPU baseline) | 0.0553 | 0.178x | 75.0s |

### Phase 4 成果总结

| 指标 | Phase 3 → Phase 4 | vs CPU Baseline |
|------|:------------------:|:---------------:|
| 离线 RTF | 0.163x → **0.079x** (2.1x 加速) | **12% 更快** |
| 流式 RTF | 0.255x → **0.156x** (1.6x 加速) | **12% 更快** |
| Decoder round-trips/层 | ~8 | **0** (prefill 层内) |
| 总 round-trips | ~224 | **2** (仅 input upload + output download + KV sync) |

Phase 4 是从 Phase 3 的"GPU 比 CPU 慢 80%"到"GPU 比 CPU 快 12%"的关键转折点。证明了**消除 GPU↔CPU round-trip 比加速单个 kernel 重要得多**。

---

## 经验总结：GPU 加速的核心原则

### 1. 数据搬运开销 > 计算加速收益

在 Apple Silicon 统一内存架构下，即使物理内存共享，candle 的 `to_device` 仍有 command buffer submission + 内存屏障的开销。**每次 GPU↔CPU 拷贝 ~0.1-0.5ms**，28 层 × 8 次 = 22-112ms，远超 GPU matmul 节省的时间。

Phase 3 vs Phase 4 的数据完美证明了这一点：
- Phase 3: 每层 8 次 round-trip → GPU 比 CPU **慢 80%**
- Phase 4: 每层 0 次 round-trip → GPU 比 CPU **快 12%**

### 2. 研读 shader 源码而非依赖 API 注释

- `call_sdpa_full` 注释说"不支持 GQA"→ 实际支持
- `call_rope_thd` Rust wrapper 无任何关于 cos/sin 布局的说明 → 必须读 `.metal` 源码才能发现 `[t, d/2]` 索引

### 3. candle Metal 的 Tensor API 与底层 kernel 有巨大 gap

candle 高层 API 缺少很多 Metal kernel 的入口（LayerNorm, RMSNorm, Softmax, RoPE, SDPA 等都没有直接支持）。通过 CustomOp trait 可以桥接这个 gap，但需要手动管理：
- `MetalStorage` 的构造（Arc\<Buffer\>, count, dtype）
- `BackpropOp::none()` 用于推理场景
- 输入/输出 tensor 的 `start_offset` × `dtype.size_in_bytes()` 转换
- storage guard 的生命周期管理（SDPA 需要同时持有 3 个 guard）

### 4. "先跑通再优化"的重要性

Phase 4 分两步验证：
1. 先实现全 GPU prefill + KV sync 到 CPU → 验证 CER 正确性
2. 后续 Phase 5 实现全 GPU decode → 消除 KV sync 开销

如果一步到位实现 prefill + decode，出 bug 时无法隔离问题。

### 5. GPU KV cache 的 stride trick

`GpuKvCache` 预分配 `[1, n_kv_heads, max_seq, head_dim]`，通过 `narrow(dim=2, 0, actual_seq_len)` 创建逻辑 view。`call_sdpa_full` 通过 strides 参数自动只读取前 `actual_seq_len` 个 position，无需复制或重新分配内存。

cache grow 时用 `blit_command_encoder` 做 GPU 端 buffer 拷贝，完全不需要下载数据。
