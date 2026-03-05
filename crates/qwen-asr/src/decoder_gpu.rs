//! GPU-accelerated decoder using candle Metal backend.
//!
//! Phase 4+5: Full GPU forward pass — RoPE, SDPA, KV cache all on Metal.
//! Eliminates GPU↔CPU round-trips within each transformer layer.

use candle_core::{Device, DType, Result, Tensor, Storage};
use candle_core::backend::BackendStorage;

use crate::config::*;
use crate::decoder::{Decoder, DecoderInt8};
use crate::decoder::{KvCache, RopeCache, DecoderBuffers};
use crate::kernels;
use crate::metal_ops::{RmsNormOp, RopeOp};

// ---------------------------------------------------------------------------
// GPU weight structs
// ---------------------------------------------------------------------------

pub struct DecLayerGpu {
    pub wq: Tensor,       // [q_dim, hidden]
    pub wk: Tensor,       // [kv_dim, hidden]
    pub wv: Tensor,       // [kv_dim, hidden]
    pub wo: Tensor,       // [hidden, q_dim]
    pub gate_up: Tensor,  // [2*intermediate, hidden]
    pub down: Tensor,     // [hidden, intermediate]
    // Norm weights on GPU for Metal native RmsNorm kernel
    pub input_norm: Tensor,     // [hidden]
    pub post_attn_norm: Tensor, // [hidden]
    pub q_norm: Tensor,         // [head_dim]
    pub k_norm: Tensor,         // [head_dim]
}

pub struct DecoderGpuWeights {
    pub layers: Vec<DecLayerGpu>,
    pub device: Device,
}

// ---------------------------------------------------------------------------
// Weight upload helpers
// ---------------------------------------------------------------------------

/// Upload a BF16 pointer to GPU as F32 tensor.
fn bf16_ptr_to_f32_gpu(ptr: *const u16, rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
    let n = rows * cols;
    let mut buf = vec![0.0f32; n];
    for i in 0..n {
        let bits = unsafe { *ptr.add(i) };
        buf[i] = f32::from_bits((bits as u32) << 16);
    }
    Tensor::from_slice(&buf, &[rows, cols], &Device::Cpu)?.to_device(dev)
}

/// Upload an owned BF16 Vec<u16> to GPU as F32 tensor.
fn bf16_vec_to_f32_gpu(data: &[u16], rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
    let n = rows * cols;
    let mut buf = vec![0.0f32; n];
    for i in 0..n {
        buf[i] = f32::from_bits((data[i] as u32) << 16);
    }
    Tensor::from_slice(&buf, &[rows, cols], &Device::Cpu)?.to_device(dev)
}

/// Upload a 1D f32 weight to GPU.
fn upload_1d(data: &[f32], len: usize, dev: &Device) -> Result<Tensor> {
    Tensor::from_slice(data, &[len], &Device::Cpu)?.to_device(dev)
}

impl DecoderGpuWeights {
    /// Upload BF16 decoder weights to Metal GPU.
    pub fn from_decoder(dec: &Decoder, cfg: &QwenConfig, device: &Device) -> Result<Self> {
        let h = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let inter = cfg.dec_intermediate;
        let head_dim = cfg.dec_head_dim;

        let mut layers = Vec::with_capacity(dec.layers.len());
        for l in &dec.layers {
            layers.push(DecLayerGpu {
                wq: bf16_ptr_to_f32_gpu(l.wq_weight_bf16, q_dim, h, device)?,
                wk: bf16_ptr_to_f32_gpu(l.wk_weight_bf16, kv_dim, h, device)?,
                wv: bf16_ptr_to_f32_gpu(l.wv_weight_bf16, kv_dim, h, device)?,
                wo: bf16_ptr_to_f32_gpu(l.wo_weight_bf16, h, q_dim, device)?,
                gate_up: bf16_vec_to_f32_gpu(&l.gate_up_fused_bf16, 2 * inter, h, device)?,
                down: bf16_ptr_to_f32_gpu(l.down_weight_bf16, h, inter, device)?,
                input_norm: upload_1d(&l.input_norm, h, device)?,
                post_attn_norm: upload_1d(&l.post_attn_norm, h, device)?,
                q_norm: upload_1d(&l.q_norm_weight, head_dim, device)?,
                k_norm: upload_1d(&l.k_norm_weight, head_dim, device)?,
            });
        }

        Ok(DecoderGpuWeights { layers, device: device.clone() })
    }

    /// Upload INT8 decoder weights (dequant to F32) to Metal GPU.
    pub fn from_decoder_int8(dec: &DecoderInt8, cfg: &QwenConfig, device: &Device) -> Result<Self> {
        let h = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let inter = cfg.dec_intermediate;
        let head_dim = cfg.dec_head_dim;

        fn dequant_upload(qw: &crate::quantize::QuantWeight, rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
            let n = rows * cols;
            let mut buf = vec![0.0f32; n];
            for o in 0..rows {
                let scale = unsafe { *qw.scales.add(o) };
                let base = o * cols;
                for k in 0..cols {
                    buf[base + k] = unsafe { *qw.data.add(base + k) } as f32 * scale;
                }
            }
            Tensor::from_slice(&buf, &[rows, cols], &Device::Cpu)?.to_device(dev)
        }

        let mut layers = Vec::with_capacity(dec.layers.len());
        for l in &dec.layers {
            // gate_up_fused is already interleaved [2*intermediate, hidden] as QuantWeightOwned
            let fused_rows = 2 * inter;
            let mut fused = vec![0.0f32; fused_rows * h];
            for r in 0..fused_rows {
                let scale = l.gate_up_fused.scales[r];
                let base = r * h;
                for k in 0..h {
                    fused[base + k] = l.gate_up_fused.data[base + k] as f32 * scale;
                }
            }
            let gate_up = Tensor::from_slice(&fused, &[fused_rows, h], &Device::Cpu)?.to_device(device)?;

            layers.push(DecLayerGpu {
                wq: dequant_upload(&l.wq, q_dim, h, device)?,
                wk: dequant_upload(&l.wk, kv_dim, h, device)?,
                wv: dequant_upload(&l.wv, kv_dim, h, device)?,
                wo: dequant_upload(&l.wo, h, q_dim, device)?,
                gate_up,
                down: dequant_upload(&l.down, h, inter, device)?,
                input_norm: upload_1d(&l.input_norm, h, device)?,
                post_attn_norm: upload_1d(&l.post_attn_norm, h, device)?,
                q_norm: upload_1d(&l.q_norm_weight, head_dim, device)?,
                k_norm: upload_1d(&l.k_norm_weight, head_dim, device)?,
            });
        }

        Ok(DecoderGpuWeights { layers, device: device.clone() })
    }
}

// ---------------------------------------------------------------------------
// GPU Rope Cache — cos/sin tables on GPU for Metal RoPE kernel
// ---------------------------------------------------------------------------

pub struct GpuRopeCache {
    pub cos: Tensor,       // [cap, head_dim] on GPU, F32
    pub sin: Tensor,       // [cap, head_dim] on GPU, F32
    pub cap: usize,        // positions allocated
    pub head_dim: usize,
    pub device: Device,
}

impl GpuRopeCache {
    pub fn new(device: &Device, head_dim: usize) -> Self {
        GpuRopeCache {
            cos: Tensor::zeros(&[1, head_dim], DType::F32, device).unwrap(),
            sin: Tensor::zeros(&[1, head_dim], DType::F32, device).unwrap(),
            cap: 0,
            head_dim,
            device: device.clone(),
        }
    }

    /// Ensure GPU rope tables cover at least `required_pos` positions.
    /// Reads from CPU RopeCache which must already have been ensured.
    /// Metal rope_thd kernel indexes cos/sin as [t, d/2], so we only upload
    /// the first half of each row from the CPU cache (which stores [t, d] with duplication).
    pub fn ensure(&mut self, required_pos: usize, cpu_rope: &RopeCache) -> Result<()> {
        if required_pos <= self.cap {
            return Ok(());
        }
        let new_cap = cpu_rope.cap;
        let hd = self.head_dim;
        let half = hd / 2;

        // Extract only the first half of each row (kernel indexes as [t, d/2])
        let mut cos_half = vec![0.0f32; new_cap * half];
        let mut sin_half = vec![0.0f32; new_cap * half];
        for pos in 0..new_cap {
            cos_half[pos * half..(pos + 1) * half]
                .copy_from_slice(&cpu_rope.cos[pos * hd..pos * hd + half]);
            sin_half[pos * half..(pos + 1) * half]
                .copy_from_slice(&cpu_rope.sin[pos * hd..pos * hd + half]);
        }

        self.cos = Tensor::from_slice(&cos_half, &[new_cap, half], &Device::Cpu)?
            .to_device(&self.device)?;
        self.sin = Tensor::from_slice(&sin_half, &[new_cap, half], &Device::Cpu)?
            .to_device(&self.device)?;
        self.cap = new_cap;
        Ok(())
    }

    /// Get cos/sin slices for positions [start_pos .. start_pos + seq_len].
    pub fn get_slice(&self, start_pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;
        Ok((cos, sin))
    }
}

// ---------------------------------------------------------------------------
// GPU KV Cache — pre-allocated buffers on GPU for all layers
// ---------------------------------------------------------------------------

pub struct GpuKvCache {
    pub k: Vec<Tensor>,   // per-layer [1, n_kv_heads, max_seq, head_dim] on GPU
    pub v: Vec<Tensor>,   // per-layer [1, n_kv_heads, max_seq, head_dim] on GPU
    pub len: usize,       // number of positions filled
    pub max_seq: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub device: Device,
}

impl GpuKvCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize, device: &Device) -> Result<Self> {
        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k.push(Tensor::zeros(&[1, n_kv_heads, max_seq, head_dim], DType::F32, device)?);
            v.push(Tensor::zeros(&[1, n_kv_heads, max_seq, head_dim], DType::F32, device)?);
        }
        Ok(GpuKvCache {
            k, v,
            len: 0,
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
            device: device.clone(),
        })
    }

    /// Write new K/V data into cache for a given layer.
    /// new_k, new_v: [1, n_kv_heads, seq_len, head_dim] contiguous GPU tensors
    pub fn write(&mut self, layer: usize, start_pos: usize, new_k: &Tensor, new_v: &Tensor, seq_len: usize) -> Result<()> {
        let needed = start_pos + seq_len;
        if needed > self.max_seq {
            self.grow(needed + 1024)?;
        }

        // Use slice_assign via Tensor API — narrow then copy
        // Actually candle doesn't have in-place scatter. We'll use a different approach:
        // Rebuild the layer's cache tensor by concatenating the unchanged prefix + new data + unused suffix
        // But this is expensive. Better approach: use Metal blit copy.
        //
        // For now, use a simpler approach: download the new K/V slices and write them into
        // the cache buffer via blit. Since candle's MetalStorage wraps an Arc<Buffer>,
        // we can access the raw buffer for blit operations.
        self.blit_write(layer, start_pos, new_k, new_v, seq_len)?;
        Ok(())
    }

    fn blit_write(&mut self, layer: usize, start_pos: usize, new_k: &Tensor, new_v: &Tensor, seq_len: usize) -> Result<()> {
        // new_k: [1, n_kv_heads, seq_len, head_dim] contiguous
        // cache_k: [1, n_kv_heads, max_seq, head_dim]
        // For each KV head, copy seq_len * head_dim floats into the right position
        let hd = self.head_dim;
        let n_kv = self.n_kv_heads;
        let max_s = self.max_seq;
        let bsz = 4usize; // f32 = 4 bytes

        // Access raw Metal buffers
        let (cache_k_guard, _) = self.k[layer].storage_and_layout();
        let (cache_v_guard, _) = self.v[layer].storage_and_layout();
        let (new_k_guard, new_k_layout) = new_k.storage_and_layout();
        let (new_v_guard, new_v_layout) = new_v.storage_and_layout();

        let cache_k_ms = match &*cache_k_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("kv_cache: expected Metal") };
        let cache_v_ms = match &*cache_v_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("kv_cache: expected Metal") };
        let new_k_ms = match &*new_k_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("kv_cache: expected Metal") };
        let new_v_ms = match &*new_v_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("kv_cache: expected Metal") };

        let device = cache_k_ms.device().clone();
        let blit = device.blit_command_encoder()?;

        let new_k_off = new_k_layout.start_offset() * bsz;
        let new_v_off = new_v_layout.start_offset() * bsz;

        for h in 0..n_kv {
            let copy_bytes = seq_len * hd * bsz;
            // Source offset: head h in new_k = h * seq_len * hd
            let src_k_off = new_k_off + h * seq_len * hd * bsz;
            let src_v_off = new_v_off + h * seq_len * hd * bsz;
            // Dest offset: head h in cache = h * max_seq * hd + start_pos * hd
            let dst_k_off = (h * max_s * hd + start_pos * hd) * bsz;
            let dst_v_off = (h * max_s * hd + start_pos * hd) * bsz;

            blit.copy_from_buffer(
                new_k_ms.buffer(), src_k_off,
                cache_k_ms.buffer(), dst_k_off,
                copy_bytes,
            );
            blit.copy_from_buffer(
                new_v_ms.buffer(), src_v_off,
                cache_v_ms.buffer(), dst_v_off,
                copy_bytes,
            );
        }

        blit.end_encoding();
        drop(cache_k_guard);
        drop(cache_v_guard);
        drop(new_k_guard);
        drop(new_v_guard);
        Ok(())
    }

    /// Get K/V tensors for attention — return views with logical shape [1, n_kv_heads, total_seq, head_dim]
    /// but using the physical stride from max_seq. SDPA reads via strides.
    pub fn get_kv_for_attention(&self, layer: usize, total_seq: usize) -> Result<(Tensor, Tensor)> {
        // We need tensors with shape [1, n_kv_heads, total_seq, head_dim]
        // but strides [n_kv_heads * max_seq * hd, max_seq * hd, hd, 1]
        // This is just a narrow on dim=2 of the full [1, n_kv, max_seq, hd] cache tensor
        let k = self.k[layer].narrow(2, 0, total_seq)?;
        let v = self.v[layer].narrow(2, 0, total_seq)?;
        Ok((k, v))
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut new_max = self.max_seq;
        while new_max < required {
            new_max *= 2;
        }

        let hd = self.head_dim;
        let n_kv = self.n_kv_heads;
        let old_max = self.max_seq;

        for layer in 0..self.n_layers {
            let new_k = Tensor::zeros(&[1, n_kv, new_max, hd], DType::F32, &self.device)?;
            let new_v = Tensor::zeros(&[1, n_kv, new_max, hd], DType::F32, &self.device)?;

            if self.len > 0 {
                // Blit old data into new buffers
                let bsz = 4usize;
                let (old_k_guard, _) = self.k[layer].storage_and_layout();
                let (new_k_guard, _) = new_k.storage_and_layout();
                let (old_v_guard, _) = self.v[layer].storage_and_layout();
                let (new_v_guard, _) = new_v.storage_and_layout();

                let old_k_ms = match &*old_k_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("grow: expected Metal") };
                let new_k_ms = match &*new_k_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("grow: expected Metal") };
                let old_v_ms = match &*old_v_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("grow: expected Metal") };
                let new_v_ms = match &*new_v_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("grow: expected Metal") };

                let device = old_k_ms.device().clone();
                let blit = device.blit_command_encoder()?;

                for h in 0..n_kv {
                    let copy_bytes = self.len * hd * bsz;
                    let old_off = h * old_max * hd * bsz;
                    let new_off = h * new_max * hd * bsz;
                    blit.copy_from_buffer(old_k_ms.buffer(), old_off, new_k_ms.buffer(), new_off, copy_bytes);
                    blit.copy_from_buffer(old_v_ms.buffer(), old_off, new_v_ms.buffer(), new_off, copy_bytes);
                }

                blit.end_encoding();
                drop(old_k_guard);
                drop(new_k_guard);
                drop(old_v_guard);
                drop(new_v_guard);
            }

            self.k[layer] = new_k;
            self.v[layer] = new_v;
        }

        self.max_seq = new_max;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Download GPU KV cache data to CPU KV cache for CPU decode fallback.
    pub fn sync_to_cpu(&self, cpu_kv: &mut KvCache) -> Result<()> {
        let total_seq = self.len;
        if total_seq == 0 { return Ok(()); }

        let needed = total_seq;
        if needed > cpu_kv.max_seq {
            cpu_kv.grow(needed + 1024);
        }

        let hd = self.head_dim;
        let n_kv = self.n_kv_heads;

        for layer in 0..self.n_layers {
            // Download K and V for this layer: [1, n_kv_heads, total_seq, head_dim]
            let k_view = self.k[layer].narrow(2, 0, total_seq)?;
            let v_view = self.v[layer].narrow(2, 0, total_seq)?;

            // contiguous() + download
            let k_data = k_view.contiguous()?.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let v_data = v_view.contiguous()?.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

            // k_data layout: [n_kv_heads, total_seq, head_dim] (batch dim squeezed by flatten)
            // CPU KV cache layout: [total_seq, kv_dim] per layer, where kv_dim = n_kv_heads * head_dim
            // (stored as contiguous rows of kv_dim)
            let kv_dim = n_kv * hd;
            for s in 0..total_seq {
                for h in 0..n_kv {
                    let gpu_off = h * total_seq * hd + s * hd;
                    let cpu_off = (layer * cpu_kv.max_seq + s) * kv_dim + h * hd;
                    cpu_kv.k[cpu_off..cpu_off + hd]
                        .copy_from_slice(&k_data[gpu_off..gpu_off + hd]);
                    cpu_kv.v[cpu_off..cpu_off + hd]
                        .copy_from_slice(&v_data[gpu_off..gpu_off + hd]);
                }
            }
        }

        cpu_kv.len = total_seq;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPU-accelerated prefill
// ---------------------------------------------------------------------------

/// GPU linear: y = x @ W^T, where W is [out_dim, in_dim] and x is [seq, in_dim].
/// Returns [seq, out_dim] on GPU.
fn gpu_linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)
}

/// GPU RMSNorm via Metal native kernel (CustomOp2).
fn rms_norm_gpu(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op2_no_bwd(w, &RmsNormOp { eps })
}

/// GPU per-head RMSNorm: reshape [seq, n_heads*head_dim] → [seq*n_heads, head_dim],
/// apply RMSNorm with per-head weights, then reshape back.
fn per_head_rms_norm_gpu(x: &Tensor, w: &Tensor, seq: usize, n_heads: usize, head_dim: usize, eps: f32) -> Result<Tensor> {
    let flat = x.reshape(&[seq * n_heads, head_dim])?;
    let normed = flat.apply_op2_no_bwd(w, &RmsNormOp { eps })?;
    normed.reshape(&[seq, n_heads * head_dim])
}

/// GPU SwiGLU: gate_up is [seq, 2*intermediate] (interleaved gate,up pairs).
/// Returns [seq, intermediate].
fn swiglu_gpu(gate_up: &Tensor, seq: usize, intermediate: usize) -> Result<Tensor> {
    // gate_up layout: for each row, [g0, u0, g1, u1, ..., g_{I-1}, u_{I-1}]
    let reshaped = gate_up.reshape(&[seq, intermediate, 2])?;
    let gate = reshaped.narrow(2, 0, 1)?.reshape(&[seq, intermediate])?;
    let up = reshaped.narrow(2, 1, 1)?.reshape(&[seq, intermediate])?;
    let activated = gate.silu()?;
    activated.mul(&up)
}

/// GPU RoPE via Metal native kernel (CustomOp3).
/// x: flat [seq * n_heads * head_dim] on GPU
/// cos, sin: [seq, head_dim/2] on GPU (Metal rope_thd layout)
fn rope_gpu(x: &Tensor, cos: &Tensor, sin: &Tensor, n_heads: usize, head_dim: usize) -> Result<Tensor> {
    x.apply_op3_no_bwd(cos, sin, &RopeOp { n_heads, head_dim })
}

// ---------------------------------------------------------------------------
// Full GPU prefill (Phase 4) — all ops on Metal, no per-layer round-trips
// ---------------------------------------------------------------------------

/// Full GPU decoder prefill: RoPE, SDPA, KV cache all on Metal.
/// Falls back to partial-GPU prefill on error.
pub fn decoder_prefill_full_gpu(
    gpu_weights: &DecoderGpuWeights,
    cfg: &QwenConfig,
    gpu_kv: &mut GpuKvCache,
    gpu_rope: &mut GpuRopeCache,
    cpu_kv: &mut KvCache,
    cpu_rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) {
    if let Err(e) = decoder_prefill_full_gpu_inner(
        gpu_weights, cfg, gpu_kv, gpu_rope, cpu_rope, bufs, input_embeds, seq_len
    ) {
        eprintln!("[metal] full GPU prefill failed: {}, falling back to CPU", e);
        // Sync GPU KV cache → CPU KV cache is not straightforward, so reset and re-prefill on CPU.
        // In practice, this fallback should very rarely trigger.
        gpu_kv.reset();
        // CPU fallback: caller should handle
    } else {
        // Sync GPU KV cache → CPU for decode (decode still runs on CPU for better perf).
        if let Err(e) = gpu_kv.sync_to_cpu(cpu_kv) {
            eprintln!("[metal] GPU→CPU KV sync failed: {}", e);
        }
        cpu_kv.len = gpu_kv.len;
    }
}

fn decoder_prefill_full_gpu_inner(
    gpu_weights: &DecoderGpuWeights,
    cfg: &QwenConfig,
    gpu_kv: &mut GpuKvCache,
    gpu_rope: &mut GpuRopeCache,
    cpu_rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) -> Result<()> {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let dev = &gpu_weights.device;

    let start_pos = gpu_kv.len;

    // Ensure rope caches
    cpu_rope.ensure(start_pos + seq_len, head_dim, theta);
    gpu_rope.ensure(start_pos + seq_len, cpu_rope)?;
    let (rope_cos, rope_sin) = gpu_rope.get_slice(start_pos, seq_len)?;

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Upload input embeddings to GPU once
    let mut x_gpu = Tensor::from_slice(&input_embeds[..seq_len * dim], &[seq_len, dim], &Device::Cpu)?
        .to_device(dev)?;

    for (layer_idx, gpu_layer) in gpu_weights.layers.iter().enumerate() {
        // --- RMSNorm ---
        let x_norm = rms_norm_gpu(&x_gpu, &gpu_layer.input_norm, eps)?;

        // --- Q/K/V Linear ---
        let q_flat = gpu_linear(&x_norm, &gpu_layer.wq)?;
        let k_flat = gpu_linear(&x_norm, &gpu_layer.wk)?;
        let v_flat = gpu_linear(&x_norm, &gpu_layer.wv)?;

        // --- Per-head RMSNorm ---
        let q_normed = per_head_rms_norm_gpu(&q_flat, &gpu_layer.q_norm, seq_len, n_heads, head_dim, eps)?;
        let k_normed = per_head_rms_norm_gpu(&k_flat, &gpu_layer.k_norm, seq_len, n_kv_heads, head_dim, eps)?;

        // --- RoPE on GPU ---
        let q_roped = rope_gpu(&q_normed, &rope_cos, &rope_sin, n_heads, head_dim)?;
        let k_roped = rope_gpu(&k_normed, &rope_cos, &rope_sin, n_kv_heads, head_dim)?;

        // --- Reshape for attention ---
        // q: [seq, q_dim] → [1, n_heads, seq, head_dim]
        let q_4d = q_roped.reshape(&[seq_len, n_heads, head_dim])?.permute((1, 0, 2))?.unsqueeze(0)?;
        // k, v: [seq, kv_dim] → [1, n_kv_heads, seq, head_dim]
        let k_4d = k_roped.reshape(&[seq_len, n_kv_heads, head_dim])?.permute((1, 0, 2))?.unsqueeze(0)?;
        let v_4d = v_flat.reshape(&[seq_len, n_kv_heads, head_dim])?.permute((1, 0, 2))?.unsqueeze(0)?;

        // Make k_4d and v_4d contiguous for blit write
        let k_4d = k_4d.contiguous()?;
        let v_4d = v_4d.contiguous()?;

        // --- Write K/V to GPU cache ---
        gpu_kv.write(layer_idx, start_pos, &k_4d, &v_4d, seq_len)?;

        // --- SDPA ---
        let total_seq = start_pos + seq_len;
        let (cache_k, cache_v) = gpu_kv.get_kv_for_attention(layer_idx, total_seq)?;

        let attn_out = crate::metal_ops::sdpa_full_gpu(&q_4d, &cache_k, &cache_v, scale)?;

        // attn_out: [1, n_heads, seq, head_dim] → [seq, q_dim]
        let attn_2d = attn_out.squeeze(0)?.permute((1, 0, 2))?.reshape(&[seq_len, q_dim])?.contiguous()?;

        // --- O projection + residual ---
        let proj = gpu_linear(&attn_2d, &gpu_layer.wo)?;
        x_gpu = (x_gpu + proj)?;

        // --- Post-attention RMSNorm ---
        let x_norm2 = rms_norm_gpu(&x_gpu, &gpu_layer.post_attn_norm, eps)?;

        // --- FFN ---
        let gate_up = gpu_linear(&x_norm2, &gpu_layer.gate_up)?;
        let swiglu_out = swiglu_gpu(&gate_up, seq_len, intermediate)?;
        let ffn_out = gpu_linear(&swiglu_out, &gpu_layer.down)?;
        x_gpu = (x_gpu + ffn_out)?;
    }

    // Download final hidden state to CPU
    let x_cpu = x_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
        .flatten_all()?.to_vec1::<f32>()?;

    bufs.ensure_prefill(seq_len, cfg);
    bufs.pref_x[..seq_len * dim].copy_from_slice(&x_cpu);

    gpu_kv.len = start_pos + seq_len;
    Ok(())
}

// ---------------------------------------------------------------------------
// Full GPU single-token decode (Phase 5)
// ---------------------------------------------------------------------------

/// Full GPU single-token decoder forward pass.
///
/// `input_embed`: [dim] f32 slice — the embedding of the current token.
/// Returns the final hidden state as [dim] f32 Vec (caller does LM head argmax on CPU).
pub fn decoder_forward_gpu(
    gpu_weights: &DecoderGpuWeights,
    cfg: &QwenConfig,
    gpu_kv: &mut GpuKvCache,
    gpu_rope: &mut GpuRopeCache,
    cpu_rope: &mut RopeCache,
    input_embed: &[f32],
) -> Result<Vec<f32>> {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let dev = &gpu_weights.device;

    let pos = gpu_kv.len;

    // Ensure rope caches cover pos+1
    cpu_rope.ensure(pos + 1, head_dim, theta);
    gpu_rope.ensure(pos + 1, cpu_rope)?;
    let (rope_cos, rope_sin) = gpu_rope.get_slice(pos, 1)?;

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Upload input embedding to GPU: [1, dim]
    let mut x_gpu = Tensor::from_slice(&input_embed[..dim], &[1, dim], &Device::Cpu)?
        .to_device(dev)?;

    for (layer_idx, gpu_layer) in gpu_weights.layers.iter().enumerate() {
        // RMSNorm
        let x_norm = rms_norm_gpu(&x_gpu, &gpu_layer.input_norm, eps)?;

        // Q/K/V Linear
        let q_flat = gpu_linear(&x_norm, &gpu_layer.wq)?;
        let k_flat = gpu_linear(&x_norm, &gpu_layer.wk)?;
        let v_flat = gpu_linear(&x_norm, &gpu_layer.wv)?;

        // Per-head RMSNorm
        let q_normed = per_head_rms_norm_gpu(&q_flat, &gpu_layer.q_norm, 1, n_heads, head_dim, eps)?;
        let k_normed = per_head_rms_norm_gpu(&k_flat, &gpu_layer.k_norm, 1, n_kv_heads, head_dim, eps)?;

        // RoPE on GPU
        let q_roped = rope_gpu(&q_normed, &rope_cos, &rope_sin, n_heads, head_dim)?;
        let k_roped = rope_gpu(&k_normed, &rope_cos, &rope_sin, n_kv_heads, head_dim)?;

        // Reshape for attention: [1, n_heads, 1, head_dim]
        let q_4d = q_roped.reshape(&[1, n_heads, head_dim])?.permute((1, 0, 2))?.unsqueeze(0)?;
        let k_4d = k_roped.reshape(&[1, n_kv_heads, head_dim])?.permute((1, 0, 2))?.unsqueeze(0)?;
        let v_4d = v_flat.reshape(&[1, n_kv_heads, head_dim])?.permute((1, 0, 2))?.unsqueeze(0)?;

        // Ensure contiguous layout for blit write and SDPA
        let q_4d = q_4d.contiguous()?;
        let k_4d = k_4d.contiguous()?;
        let v_4d = v_4d.contiguous()?;

        // Write K/V to GPU cache
        gpu_kv.write(layer_idx, pos, &k_4d, &v_4d, 1)?;

        // SDPA — use sdpa_full for single-token decode too.
        // Note: sdpa_vector produces NaN with GQA + non-contiguous KV cache strides
        // in candle-metal-kernels 0.9.2. sdpa_full handles this correctly.
        let total_seq = pos + 1;
        let (cache_k, cache_v) = gpu_kv.get_kv_for_attention(layer_idx, total_seq)?;
        let attn_out = crate::metal_ops::sdpa_full_gpu(&q_4d, &cache_k, &cache_v, scale)?;

        // attn_out: [1, n_heads, 1, head_dim] → [1, q_dim]
        let attn_2d = attn_out.squeeze(0)?.permute((1, 0, 2))?.reshape(&[1, q_dim])?.contiguous()?;

        // O projection + residual
        let proj = gpu_linear(&attn_2d, &gpu_layer.wo)?;
        x_gpu = (x_gpu + proj)?;

        // Post-attention RMSNorm
        let x_norm2 = rms_norm_gpu(&x_gpu, &gpu_layer.post_attn_norm, eps)?;

        // FFN
        let gate_up = gpu_linear(&x_norm2, &gpu_layer.gate_up)?;
        let swiglu_out = swiglu_gpu(&gate_up, 1, intermediate)?;
        let ffn_out = gpu_linear(&swiglu_out, &gpu_layer.down)?;
        x_gpu = (x_gpu + ffn_out)?;
    }

    gpu_kv.len = pos + 1;

    // Download final hidden state to CPU: [1, dim] → Vec<f32>
    let x_cpu = x_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
        .flatten_all()?.to_vec1::<f32>()?;

    Ok(x_cpu)
}

// ---------------------------------------------------------------------------
// Old partial-GPU prefill (Phase 3 fallback)
// ---------------------------------------------------------------------------
pub fn decoder_prefill_gpu(
    gpu_weights: &DecoderGpuWeights,
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) {
    if let Err(e) = decoder_prefill_gpu_inner(
        gpu_weights, decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len
    ) {
        eprintln!("[metal] decoder_prefill_gpu failed: {}, falling back to CPU", e);
        crate::decoder::decoder_prefill(decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len);
    }
}

fn decoder_prefill_gpu_inner(
    gpu_weights: &DecoderGpuWeights,
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) -> Result<()> {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dev = &gpu_weights.device;

    // Ensure KV cache
    let needed = kv_cache.len + seq_len;
    if needed > kv_cache.max_seq {
        kv_cache.grow(needed + 1024);
    }

    bufs.ensure_prefill(seq_len, cfg);

    let x = &mut bufs.pref_x[..seq_len * dim];
    x.copy_from_slice(&input_embeds[..seq_len * dim]);

    let start_pos = kv_cache.len;
    rope.ensure(start_pos + seq_len, head_dim, theta);
    let rope_cos = rope.cos_range(start_pos, seq_len);
    let rope_sin = rope.sin_range(start_pos, seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, (_layer, gpu_layer)) in decoder.layers.iter().zip(gpu_weights.layers.iter()).enumerate() {
        // --- RMSNorm on GPU ---
        let x_gpu = Tensor::from_slice(&bufs.pref_x[..seq_len * dim], &[seq_len, dim], &Device::Cpu)?.to_device(dev)?;
        let x_norm_gpu = rms_norm_gpu(&x_gpu, &gpu_layer.input_norm, eps)?;

        // --- Q/K/V Linear on GPU ---
        let q_gpu = gpu_linear(&x_norm_gpu, &gpu_layer.wq)?;
        let k_gpu = gpu_linear(&x_norm_gpu, &gpu_layer.wk)?;
        let v_gpu = gpu_linear(&x_norm_gpu, &gpu_layer.wv)?;

        // --- Per-head RMSNorm on GPU ---
        let q_normed = per_head_rms_norm_gpu(&q_gpu, &gpu_layer.q_norm, seq_len, n_heads, head_dim, eps)?;
        let k_normed = per_head_rms_norm_gpu(&k_gpu, &gpu_layer.k_norm, seq_len, n_kv_heads, head_dim, eps)?;

        // Download Q/K (normed) for RoPE + attention, V directly
        let q_cpu = q_normed.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
            .flatten_all()?.to_vec1::<f32>()?;
        let k_cpu = k_normed.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
            .flatten_all()?.to_vec1::<f32>()?;
        let v_cpu = v_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
            .flatten_all()?.to_vec1::<f32>()?;

        let q = &mut bufs.pref_q[..seq_len * q_dim];
        let k = &mut bufs.pref_k[..seq_len * kv_dim];
        let v = &mut bufs.pref_v[..seq_len * kv_dim];
        q.copy_from_slice(&q_cpu);
        k.copy_from_slice(&k_cpu);
        v.copy_from_slice(&v_cpu);

        // --- RoPE on CPU (trig functions, cheap) ---
        kernels::apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        kernels::apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        // --- Store K/V in cache ---
        for s in 0..seq_len {
            kv_cache.k_at(layer_idx, start_pos + s)
                .copy_from_slice(&bufs.pref_k[s * kv_dim..(s + 1) * kv_dim]);
            kv_cache.v_at(layer_idx, start_pos + s)
                .copy_from_slice(&bufs.pref_v[s * kv_dim..(s + 1) * kv_dim]);
        }

        // --- Causal attention on CPU ---
        let total_seq = start_pos + seq_len;
        let full_k = kv_cache.k_layer_full(layer_idx, total_seq);
        let full_v = kv_cache.v_layer_full(layer_idx, total_seq);

        let attn_out = &mut bufs.pref_attn_out[..seq_len * q_dim];
        kernels::causal_attention(attn_out, q, full_k, full_v,
                                 seq_len, total_seq, n_heads, n_kv_heads,
                                 head_dim, scale, start_pos);

        // --- Output projection on GPU + residual ---
        let attn_gpu = Tensor::from_slice(attn_out, &[seq_len, q_dim], &Device::Cpu)?.to_device(dev)?;
        let proj_gpu = gpu_linear(&attn_gpu, &gpu_layer.wo)?;
        let proj_cpu = proj_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
            .flatten_all()?.to_vec1::<f32>()?;

        let proj_out = &mut bufs.pref_proj_out[..seq_len * dim];
        proj_out.copy_from_slice(&proj_cpu);
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], proj_out, seq_len * dim);

        // --- Post-attention RMSNorm on GPU ---
        let x_gpu2 = Tensor::from_slice(&bufs.pref_x[..seq_len * dim], &[seq_len, dim], &Device::Cpu)?.to_device(dev)?;
        let x_norm2_gpu = rms_norm_gpu(&x_gpu2, &gpu_layer.post_attn_norm, eps)?;

        // --- FFN gate_up on GPU ---
        let gate_up_gpu = gpu_linear(&x_norm2_gpu, &gpu_layer.gate_up)?;

        // --- SwiGLU on GPU ---
        let swiglu_out = swiglu_gpu(&gate_up_gpu, seq_len, intermediate)?;

        // --- Down projection on GPU ---
        let ffn_out_gpu = gpu_linear(&swiglu_out, &gpu_layer.down)?;
        let ffn_out_cpu = ffn_out_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
            .flatten_all()?.to_vec1::<f32>()?;

        let ffn_out = &mut bufs.pref_ffn_out[..seq_len * dim];
        ffn_out.copy_from_slice(&ffn_out_cpu);
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], ffn_out, seq_len * dim);
    }

    kv_cache.len = start_pos + seq_len;
    Ok(())
}

/// INT8 decoder prefill on GPU (same strategy, different weight source).
pub fn decoder_prefill_int8_gpu(
    gpu_weights: &DecoderGpuWeights,
    decoder: &DecoderInt8,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) {
    if let Err(e) = decoder_prefill_int8_gpu_inner(
        gpu_weights, decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len
    ) {
        eprintln!("[metal] decoder_prefill_int8_gpu failed: {}, falling back to CPU", e);
        crate::decoder::decoder_prefill_int8(decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len);
    }
}

fn decoder_prefill_int8_gpu_inner(
    gpu_weights: &DecoderGpuWeights,
    decoder: &DecoderInt8,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) -> Result<()> {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dev = &gpu_weights.device;

    let needed = kv_cache.len + seq_len;
    if needed > kv_cache.max_seq {
        kv_cache.grow(needed + 1024);
    }

    bufs.ensure_prefill(seq_len, cfg);

    let x = &mut bufs.pref_x[..seq_len * dim];
    x.copy_from_slice(&input_embeds[..seq_len * dim]);

    let start_pos = kv_cache.len;
    rope.ensure(start_pos + seq_len, head_dim, theta);
    let rope_cos = rope.cos_range(start_pos, seq_len);
    let rope_sin = rope.sin_range(start_pos, seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, (_layer, gpu_layer)) in decoder.layers.iter().zip(gpu_weights.layers.iter()).enumerate() {
        // --- RMSNorm on GPU ---
        let x_gpu = Tensor::from_slice(&bufs.pref_x[..seq_len * dim], &[seq_len, dim], &Device::Cpu)?.to_device(dev)?;
        let x_norm_gpu = rms_norm_gpu(&x_gpu, &gpu_layer.input_norm, eps)?;

        // --- Q/K/V on GPU ---
        let q_gpu = gpu_linear(&x_norm_gpu, &gpu_layer.wq)?;
        let k_gpu = gpu_linear(&x_norm_gpu, &gpu_layer.wk)?;
        let v_gpu = gpu_linear(&x_norm_gpu, &gpu_layer.wv)?;

        // --- Per-head RMSNorm on GPU ---
        let q_normed = per_head_rms_norm_gpu(&q_gpu, &gpu_layer.q_norm, seq_len, n_heads, head_dim, eps)?;
        let k_normed = per_head_rms_norm_gpu(&k_gpu, &gpu_layer.k_norm, seq_len, n_kv_heads, head_dim, eps)?;

        // Download Q/K (normed) and V to CPU for RoPE + attention
        let q_cpu = q_normed.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let k_cpu = k_normed.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v_cpu = v_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let q = &mut bufs.pref_q[..seq_len * q_dim];
        let k = &mut bufs.pref_k[..seq_len * kv_dim];
        let v = &mut bufs.pref_v[..seq_len * kv_dim];
        q.copy_from_slice(&q_cpu);
        k.copy_from_slice(&k_cpu);
        v.copy_from_slice(&v_cpu);

        // --- RoPE on CPU ---
        kernels::apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        kernels::apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        for s in 0..seq_len {
            kv_cache.k_at(layer_idx, start_pos + s)
                .copy_from_slice(&bufs.pref_k[s * kv_dim..(s + 1) * kv_dim]);
            kv_cache.v_at(layer_idx, start_pos + s)
                .copy_from_slice(&bufs.pref_v[s * kv_dim..(s + 1) * kv_dim]);
        }

        // --- Causal attention on CPU ---
        let total_seq = start_pos + seq_len;
        let full_k = kv_cache.k_layer_full(layer_idx, total_seq);
        let full_v = kv_cache.v_layer_full(layer_idx, total_seq);

        let attn_out = &mut bufs.pref_attn_out[..seq_len * q_dim];
        kernels::causal_attention(attn_out, q, full_k, full_v,
                                 seq_len, total_seq, n_heads, n_kv_heads,
                                 head_dim, scale, start_pos);

        // --- Output projection on GPU + residual ---
        let attn_gpu = Tensor::from_slice(attn_out, &[seq_len, q_dim], &Device::Cpu)?.to_device(dev)?;
        let proj_gpu = gpu_linear(&attn_gpu, &gpu_layer.wo)?;
        let proj_cpu = proj_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let proj_out = &mut bufs.pref_proj_out[..seq_len * dim];
        proj_out.copy_from_slice(&proj_cpu);
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], proj_out, seq_len * dim);

        // --- Post-attention RMSNorm on GPU ---
        let x_gpu2 = Tensor::from_slice(&bufs.pref_x[..seq_len * dim], &[seq_len, dim], &Device::Cpu)?.to_device(dev)?;
        let x_norm2_gpu = rms_norm_gpu(&x_gpu2, &gpu_layer.post_attn_norm, eps)?;

        // --- FFN gate_up + SwiGLU on GPU ---
        let gate_up_gpu = gpu_linear(&x_norm2_gpu, &gpu_layer.gate_up)?;
        let swiglu_out = swiglu_gpu(&gate_up_gpu, seq_len, intermediate)?;

        // --- Down projection on GPU ---
        let ffn_out_gpu = gpu_linear(&swiglu_out, &gpu_layer.down)?;
        let ffn_out_cpu = ffn_out_gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let ffn_out = &mut bufs.pref_ffn_out[..seq_len * dim];
        ffn_out.copy_from_slice(&ffn_out_cpu);
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], ffn_out, seq_len * dim);
    }

    kv_cache.len = start_pos + seq_len;
    Ok(())
}
