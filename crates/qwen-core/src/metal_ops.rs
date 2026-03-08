//! CustomOp wrappers for candle-metal-kernels' native Metal shaders.
//!
//! These bypass candle's high-level Tensor API (which lacks layer_norm/rms_norm/softmax
//! methods for Metal) and dispatch directly to the optimized Metal compute kernels.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor};
use candle_core::op::BackpropOp;

// ---------------------------------------------------------------------------
// SoftmaxLastOp — softmax along last dimension
// ---------------------------------------------------------------------------

pub struct SoftmaxLastOp;

impl CustomOp1 for SoftmaxLastOp {
    fn name(&self) -> &'static str {
        "softmax_last_metal"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let shape = layout.shape();
        let dims = shape.dims();
        let last_dim = *dims.last().unwrap();
        let n_rows = shape.elem_count() / last_dim;

        let src = match storage {
            CpuStorage::F32(data) => data,
            _ => candle_core::bail!("softmax_last: only F32 supported"),
        };

        let src_offset = layout.start_offset();
        let mut out = vec![0.0f32; shape.elem_count()];

        for row in 0..n_rows {
            let base = src_offset + row * last_dim;
            let row_data = &src[base..base + last_dim];
            let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..last_dim {
                let e = (row_data[j] - max_val).exp();
                out[row * last_dim + j] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for j in 0..last_dim {
                out[row * last_dim + j] *= inv;
            }
        }

        Ok((CpuStorage::F32(out), shape.clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device().clone();
        let shape = layout.shape();
        let length = shape.elem_count();
        let last_dim = *shape.dims().last().unwrap();

        let output = device.new_buffer(length, storage.dtype(), "softmax_out")?;

        let encoder = device.command_encoder()?;
        let kernel_name = match storage.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            dt => candle_core::bail!("softmax_last: unsupported dtype {dt:?}"),
        };

        candle_metal_kernels::call_last_softmax(
            device.device(),
            &encoder,
            device.kernels(),
            kernel_name,
            length,
            last_dim,
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok((MetalStorage::new(output, device, length, storage.dtype()), shape.clone()))
    }
}

// ---------------------------------------------------------------------------
// LayerNormOp — layer normalization with scale (alpha) and bias (beta)
// ---------------------------------------------------------------------------

pub struct LayerNormOp {
    pub eps: f32,
}

impl CustomOp3 for LayerNormOp {
    fn name(&self) -> &'static str {
        "layer_norm_metal"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let shape = l1.shape();
        let dims = shape.dims();
        let dim = *dims.last().unwrap();
        let n_rows = shape.elem_count() / dim;

        let x = match s1 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("layer_norm: only F32 supported"),
        };
        let alpha = match s2 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("layer_norm: only F32 supported"),
        };
        let beta = match s3 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("layer_norm: only F32 supported"),
        };

        let x_off = l1.start_offset();
        let a_off = l2.start_offset();
        let b_off = l3.start_offset();
        let eps = self.eps;

        let mut out = vec![0.0f32; shape.elem_count()];
        for row in 0..n_rows {
            let base = x_off + row * dim;
            let row_data = &x[base..base + dim];
            let mean: f32 = row_data.iter().sum::<f32>() / dim as f32;
            let var: f32 = row_data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let out_base = row * dim;
            for j in 0..dim {
                out[out_base + j] = (row_data[j] - mean) * inv_std * alpha[a_off + j] + beta[b_off + j];
            }
        }

        Ok((CpuStorage::F32(out), shape.clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device().clone();
        let shape = l1.shape();
        let length = shape.elem_count();
        let elements_to_sum = *shape.dims().last().unwrap();

        let output = device.new_buffer(length, s1.dtype(), "layer_norm_out")?;

        let encoder = device.command_encoder()?;
        let kernel_name = match s1.dtype() {
            DType::F32 => "layernorm_f32",
            DType::F16 => "layernorm_f16",
            DType::BF16 => "layernorm_bf16",
            dt => candle_core::bail!("layer_norm: unsupported dtype {dt:?}"),
        };

        candle_metal_kernels::call_layer_norm(
            device.device(),
            &encoder,
            device.kernels(),
            kernel_name,
            length,
            elements_to_sum,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok((MetalStorage::new(output, device, length, s1.dtype()), shape.clone()))
    }
}

// ---------------------------------------------------------------------------
// RmsNormOp — RMS normalization with scale (alpha)
// ---------------------------------------------------------------------------

pub struct RmsNormOp {
    pub eps: f32,
}

impl CustomOp2 for RmsNormOp {
    fn name(&self) -> &'static str {
        "rms_norm_metal"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let shape = l1.shape();
        let dims = shape.dims();
        let dim = *dims.last().unwrap();
        let n_rows = shape.elem_count() / dim;

        let x = match s1 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("rms_norm: only F32 supported"),
        };
        let alpha = match s2 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("rms_norm: only F32 supported"),
        };

        let x_off = l1.start_offset();
        let a_off = l2.start_offset();
        let eps = self.eps;

        let mut out = vec![0.0f32; shape.elem_count()];
        for row in 0..n_rows {
            let base = x_off + row * dim;
            let row_data = &x[base..base + dim];
            let ss: f32 = row_data.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rms = 1.0 / (ss + eps).sqrt();
            let out_base = row * dim;
            for j in 0..dim {
                out[out_base + j] = row_data[j] * rms * alpha[a_off + j];
            }
        }

        Ok((CpuStorage::F32(out), shape.clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device().clone();
        let shape = l1.shape();
        let length = shape.elem_count();
        let elements_to_sum = *shape.dims().last().unwrap();

        let output = device.new_buffer(length, s1.dtype(), "rms_norm_out")?;

        let encoder = device.command_encoder()?;
        let kernel_name = match s1.dtype() {
            DType::F32 => "rmsnorm_f32",
            DType::F16 => "rmsnorm_f16",
            DType::BF16 => "rmsnorm_bf16",
            dt => candle_core::bail!("rms_norm: unsupported dtype {dt:?}"),
        };

        candle_metal_kernels::call_rms_norm(
            device.device(),
            &encoder,
            device.kernels(),
            kernel_name,
            length,
            elements_to_sum,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok((MetalStorage::new(output, device, length, s1.dtype()), shape.clone()))
    }
}

// ---------------------------------------------------------------------------
// RopeOp — Rotary Position Embedding (NeoX variant) via Metal kernel
// ---------------------------------------------------------------------------
// Inputs: (x, cos, sin)
//   x:   flat [seq * n_heads * head_dim]
//   cos: [seq, head_dim]   (full head_dim, not half)
//   sin: [seq, head_dim]
// Output: [seq * n_heads * head_dim] with RoPE applied

pub struct RopeOp {
    pub n_heads: usize,
    pub head_dim: usize,
}

impl CustomOp3 for RopeOp {
    fn name(&self) -> &'static str {
        "rope_thd_metal"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let shape = l1.shape();
        let total = shape.elem_count();
        let hd = self.head_dim;
        let half = hd / 2;
        let seq = total / (self.n_heads * hd);

        let x = match s1 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("rope: only F32 supported"),
        };
        let cos_vals = match s2 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("rope: only F32 supported"),
        };
        let sin_vals = match s3 {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("rope: only F32 supported"),
        };

        let x_off = l1.start_offset();
        let c_off = l2.start_offset();
        let s_off = l3.start_offset();

        // cos/sin are [seq, head_dim/2] — Metal rope_thd layout
        let mut out = x[x_off..x_off + total].to_vec();
        let hidden = self.n_heads * hd;
        for s in 0..seq {
            let c = &cos_vals[c_off + s * half..];
            let sn = &sin_vals[s_off + s * half..];
            for h in 0..self.n_heads {
                let base = s * hidden + h * hd;
                for d in 0..half {
                    let x1 = out[base + d];
                    let x2 = out[base + half + d];
                    out[base + d]        = x1 * c[d] - x2 * sn[d];
                    out[base + half + d] = x1 * sn[d] + x2 * c[d];
                }
            }
        }

        Ok((CpuStorage::F32(out), shape.clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = s1.device().clone();
        let shape = l1.shape();
        let total = shape.elem_count();
        let seq = total / (self.n_heads * self.head_dim);

        let output = device.new_buffer(total, s1.dtype(), "rope_out")?;

        let encoder = device.command_encoder()?;
        let kernel_name = match s1.dtype() {
            DType::F32 => "rope_thd_f32",
            DType::F16 => "rope_thd_f16",
            dt => candle_core::bail!("rope: unsupported dtype {dt:?}"),
        };

        candle_metal_kernels::call_rope_thd(
            device.device(),
            &encoder,
            device.kernels(),
            kernel_name,
            1,                     // b = 1
            seq,                   // t = seq_len
            self.n_heads,          // h
            self.head_dim,         // d
            0,                     // stride_b (unused for b=1)
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok((MetalStorage::new(output, device, total, s1.dtype()), shape.clone()))
    }
}

// ---------------------------------------------------------------------------
// SDPA standalone functions — operate directly on MetalStorage
// ---------------------------------------------------------------------------

/// Scaled Dot-Product Attention (full) on GPU.
///
/// q: [1, n_heads, seq_len, head_dim]   — contiguous GPU tensor
/// k: [1, n_kv_heads, total_seq, head_dim] — may have custom strides (KV cache)
/// v: [1, n_kv_heads, total_seq, head_dim] — may have custom strides (KV cache)
///
/// Returns: [1, n_heads, seq_len, head_dim]
#[cfg(feature = "metal")]
pub fn sdpa_full_gpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Hold all storage guards in the same scope
    let (q_guard, q_layout) = q.storage_and_layout();
    let (k_guard, k_layout) = k.storage_and_layout();
    let (v_guard, v_layout) = v.storage_and_layout();

    let q_ms = match &*q_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("sdpa_full: q not on Metal") };
    let k_ms = match &*k_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("sdpa_full: k not on Metal") };
    let v_ms = match &*v_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("sdpa_full: v not on Metal") };

    let device = q_ms.device().clone();
    let dtype = q.dtype();
    let bsz = dtype.size_in_bytes();

    let q_shape: Vec<usize> = q_layout.shape().dims().to_vec();
    let k_shape: Vec<usize> = k_layout.shape().dims().to_vec();
    let q_strides: Vec<usize> = q_layout.stride().to_vec();
    let k_strides: Vec<usize> = k_layout.stride().to_vec();
    let v_strides: Vec<usize> = v_layout.stride().to_vec();

    let n_heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[3];
    let out_len = n_heads * seq_len * head_dim;

    let o_shape = vec![1, n_heads, seq_len, head_dim];
    let o_strides = vec![n_heads * seq_len * head_dim, seq_len * head_dim, head_dim, 1];

    let output = device.new_buffer(out_len, dtype, "sdpa_full_out")?;

    let encoder = device.command_encoder()?;
    candle_metal_kernels::call_sdpa_full(
        device.device(),
        &encoder,
        device.kernels(),
        q_layout.start_offset() * bsz,
        &q_shape,
        &q_strides,
        q_ms.buffer(),
        k_layout.start_offset() * bsz,
        &k_shape,
        &k_strides,
        k_ms.buffer(),
        v_layout.start_offset() * bsz,
        v_ms.buffer(),
        &v_strides,
        None,  // mask_type
        None,  // mask_buffer
        None,  // m_strides
        &output,
        &o_strides,
        scale,
        true,  // do_causal
        candle_metal_kernels::SdpaDType::F32,
    )
    .map_err(candle_core::Error::wrap)?;

    // Drop guards before creating output tensor
    drop(q_guard);
    drop(k_guard);
    drop(v_guard);

    let metal_storage = MetalStorage::new(output, device, out_len, dtype);
    let storage = Storage::Metal(metal_storage);
    Ok(Tensor::from_storage(storage, o_shape, BackpropOp::none(), false))
}

/// Scaled Dot-Product Attention (vector / single-query) on GPU.
///
/// q: [1, n_heads, 1, head_dim]
/// k: [1, n_kv_heads, total_seq, head_dim] — may have custom strides (KV cache)
/// v: [1, n_kv_heads, total_seq, head_dim] — may have custom strides (KV cache)
///
/// Returns: [1, n_heads, 1, head_dim]
#[cfg(feature = "metal")]
pub fn sdpa_vector_gpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let (q_guard, q_layout) = q.storage_and_layout();
    let (k_guard, k_layout) = k.storage_and_layout();
    let (v_guard, v_layout) = v.storage_and_layout();

    let q_ms = match &*q_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("sdpa_vector: q not on Metal") };
    let k_ms = match &*k_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("sdpa_vector: k not on Metal") };
    let v_ms = match &*v_guard { Storage::Metal(ms) => ms, _ => candle_core::bail!("sdpa_vector: v not on Metal") };

    let device = q_ms.device().clone();
    let dtype = q.dtype();
    let bsz = dtype.size_in_bytes();

    let q_shape: Vec<usize> = q_layout.shape().dims().to_vec();
    let k_shape: Vec<usize> = k_layout.shape().dims().to_vec();
    let k_strides: Vec<usize> = k_layout.stride().to_vec();
    let v_strides: Vec<usize> = v_layout.stride().to_vec();

    let n_heads = q_shape[1];
    let head_dim = q_shape[3];
    let out_len = n_heads * head_dim;

    let o_shape = vec![1, n_heads, 1, head_dim];

    let output = device.new_buffer(out_len, dtype, "sdpa_vec_out")?;

    let encoder = device.command_encoder()?;
    candle_metal_kernels::call_sdpa_vector(
        device.device(),
        &encoder,
        device.kernels(),
        q_layout.start_offset() * bsz,
        &q_shape,
        q_ms.buffer(),
        k_layout.start_offset() * bsz,
        &k_shape,
        &k_strides,
        k_ms.buffer(),
        v_layout.start_offset() * bsz,
        &v_strides,
        v_ms.buffer(),
        &output,
        scale,
        0.0, // softcapping (unused)
        candle_metal_kernels::SdpaDType::F32,
    )
    .map_err(candle_core::Error::wrap)?;

    drop(q_guard);
    drop(k_guard);
    drop(v_guard);

    let metal_storage = MetalStorage::new(output, device, out_len, dtype);
    let storage = Storage::Metal(metal_storage);
    Ok(Tensor::from_storage(storage, o_shape, BackpropOp::none(), false))
}
