//! GPU encoder forward pass using candle Metal backend.
//!
//! Mirrors the CPU [`Encoder::forward`] logic but executes linear layers, GELU,
//! layer-norm, and softmax on the GPU via candle Tensors and Metal native kernels
//! (CustomOp). Conv2D stem runs on CPU (small, constant-size work).

use candle_core::{DType, Device, Result, Tensor};

use crate::config::*;
use crate::encoder::{Encoder, EncoderInt8};
use crate::kernels;
use crate::metal_ops::{LayerNormOp, SoftmaxLastOp};

// ---------------------------------------------------------------------------
// GPU weight structs
// ---------------------------------------------------------------------------

/// Per-layer GPU weights for the encoder transformer.
pub struct EncLayerGpu {
    pub wq: Tensor,
    pub wq_bias: Tensor,
    pub wk: Tensor,
    pub wk_bias: Tensor,
    pub wv: Tensor,
    pub wv_bias: Tensor,
    pub wo: Tensor,
    pub wo_bias: Tensor,
    pub attn_norm_w: Tensor,
    pub attn_norm_b: Tensor,
    pub fc1: Tensor,
    pub fc1_bias: Tensor,
    pub fc2: Tensor,
    pub fc2_bias: Tensor,
    pub ffn_norm_w: Tensor,
    pub ffn_norm_b: Tensor,
}

/// All encoder weights pre-uploaded to the Metal device.
pub struct EncoderGpuWeights {
    // Conv2D stem (kept as F32 slices — conv is done on CPU for now)
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub conv3_weight: Vec<f32>,
    pub conv3_bias: Vec<f32>,

    // Conv output projection on GPU
    pub conv_out_weight: Tensor, // [d_model, conv_proj_dim]

    // Transformer layers
    pub layers: Vec<EncLayerGpu>,

    // Post-processing
    pub ln_post_w: Tensor,
    pub ln_post_b: Tensor,
    pub proj1_w: Tensor,
    pub proj1_b: Tensor,
    pub proj2_w: Tensor,
    pub proj2_b: Tensor,

    pub device: Device,
}

// ---------------------------------------------------------------------------
// Weight upload helpers
// ---------------------------------------------------------------------------

/// Upload a 2D f32 weight to the GPU as F32 tensor.
fn upload_2d(data: &[f32], rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
    Tensor::from_slice(data, &[rows, cols], &Device::Cpu)?.to_device(dev)
}

/// Upload a 1D f32 bias to the GPU.
fn upload_1d(data: &[f32], len: usize, dev: &Device) -> Result<Tensor> {
    Tensor::from_slice(data, &[len], &Device::Cpu)?.to_device(dev)
}

impl EncoderGpuWeights {
    /// Upload BF16 (F32-stored) encoder weights to Metal.
    pub fn from_encoder(enc: &Encoder, cfg: &QwenConfig, device: &Device) -> Result<Self> {
        let d = cfg.enc_d_model;
        let ffn = cfg.enc_ffn_dim;
        let cpd = cfg.enc_conv_proj_dim;

        let conv_out_weight = upload_2d(&enc.conv_out_weight, d, cpd, device)?;

        let mut layers = Vec::with_capacity(enc.layers.len());
        for l in &enc.layers {
            layers.push(EncLayerGpu {
                wq: upload_2d(&l.wq_weight, d, d, device)?,
                wq_bias: upload_1d(&l.wq_bias, d, device)?,
                wk: upload_2d(&l.wk_weight, d, d, device)?,
                wk_bias: upload_1d(&l.wk_bias, d, device)?,
                wv: upload_2d(&l.wv_weight, d, d, device)?,
                wv_bias: upload_1d(&l.wv_bias, d, device)?,
                wo: upload_2d(&l.wo_weight, d, d, device)?,
                wo_bias: upload_1d(&l.wo_bias, d, device)?,
                attn_norm_w: upload_1d(&l.attn_norm_weight, d, device)?,
                attn_norm_b: upload_1d(&l.attn_norm_bias, d, device)?,
                fc1: upload_2d(&l.fc1_weight, ffn, d, device)?,
                fc1_bias: upload_1d(&l.fc1_bias, ffn, device)?,
                fc2: upload_2d(&l.fc2_weight, d, ffn, device)?,
                fc2_bias: upload_1d(&l.fc2_bias, d, device)?,
                ffn_norm_w: upload_1d(&l.ffn_norm_weight, d, device)?,
                ffn_norm_b: upload_1d(&l.ffn_norm_bias, d, device)?,
            });
        }

        Ok(EncoderGpuWeights {
            conv1_weight: enc.conv1_weight.clone(),
            conv1_bias: enc.conv1_bias.clone(),
            conv2_weight: enc.conv2_weight.clone(),
            conv2_bias: enc.conv2_bias.clone(),
            conv3_weight: enc.conv3_weight.clone(),
            conv3_bias: enc.conv3_bias.clone(),
            conv_out_weight,
            layers,
            ln_post_w: upload_1d(&enc.ln_post_weight, d, device)?,
            ln_post_b: upload_1d(&enc.ln_post_bias, d, device)?,
            proj1_w: upload_2d(&enc.proj1_weight, d, d, device)?,
            proj1_b: upload_1d(&enc.proj1_bias, d, device)?,
            proj2_w: upload_2d(&enc.proj2_weight, cfg.enc_output_dim, d, device)?,
            proj2_b: upload_1d(&enc.proj2_bias, cfg.enc_output_dim, device)?,
            device: device.clone(),
        })
    }

    /// Upload INT8 encoder weights: dequant to F32 then upload.
    pub fn from_encoder_int8(enc: &EncoderInt8, cfg: &QwenConfig, device: &Device) -> Result<Self> {
        let d = cfg.enc_d_model;
        let ffn = cfg.enc_ffn_dim;
        let cpd = cfg.enc_conv_proj_dim;

        // Dequant helper
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

        let conv_out_weight = dequant_upload(&enc.conv_out, d, cpd, device)?;

        let mut layers = Vec::with_capacity(enc.layers.len());
        for l in &enc.layers {
            layers.push(EncLayerGpu {
                wq: dequant_upload(&l.wq, d, d, device)?,
                wq_bias: upload_1d(&l.wq_bias, d, device)?,
                wk: dequant_upload(&l.wk, d, d, device)?,
                wk_bias: upload_1d(&l.wk_bias, d, device)?,
                wv: dequant_upload(&l.wv, d, d, device)?,
                wv_bias: upload_1d(&l.wv_bias, d, device)?,
                wo: dequant_upload(&l.wo, d, d, device)?,
                wo_bias: upload_1d(&l.wo_bias, d, device)?,
                attn_norm_w: upload_1d(&l.attn_norm_weight, d, device)?,
                attn_norm_b: upload_1d(&l.attn_norm_bias, d, device)?,
                fc1: dequant_upload(&l.fc1, ffn, d, device)?,
                fc1_bias: upload_1d(&l.fc1_bias, ffn, device)?,
                fc2: dequant_upload(&l.fc2, d, ffn, device)?,
                fc2_bias: upload_1d(&l.fc2_bias, d, device)?,
                ffn_norm_w: upload_1d(&l.ffn_norm_weight, d, device)?,
                ffn_norm_b: upload_1d(&l.ffn_norm_bias, d, device)?,
            });
        }

        Ok(EncoderGpuWeights {
            conv1_weight: enc.conv1_weight.clone(),
            conv1_bias: enc.conv1_bias.clone(),
            conv2_weight: enc.conv2_weight.clone(),
            conv2_bias: enc.conv2_bias.clone(),
            conv3_weight: enc.conv3_weight.clone(),
            conv3_bias: enc.conv3_bias.clone(),
            conv_out_weight,
            layers,
            ln_post_w: upload_1d(&enc.ln_post_weight, d, device)?,
            ln_post_b: upload_1d(&enc.ln_post_bias, d, device)?,
            proj1_w: dequant_upload(&enc.proj1, d, d, device)?,
            proj1_b: upload_1d(&enc.proj1_bias, d, device)?,
            proj2_w: dequant_upload(&enc.proj2, cfg.enc_output_dim, d, device)?,
            proj2_b: upload_1d(&enc.proj2_bias, cfg.enc_output_dim, device)?,
            device: device.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// GPU layer-norm / softmax (Metal native kernels via CustomOp)
// ---------------------------------------------------------------------------

/// Layer normalization via Metal native kernel (CustomOp3).
fn layer_norm_gpu(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op3_no_bwd(w, b, &LayerNormOp { eps })
}

/// Linear layer: y = x @ W^T + bias.  W is [out, in], x is [seq, in].
fn linear(x: &Tensor, w: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)?.broadcast_add(bias)
}

/// Linear layer without bias.
fn linear_nobias(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)
}

/// Softmax along the last dimension via Metal native kernel (CustomOp1).
fn softmax_gpu(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&SoftmaxLastOp)
}

/// Windowed bidirectional attention on GPU using candle tensors.
///
/// Splits seq into windows, runs standard (non-causal) scaled dot-product
/// attention within each window, then reassembles.
fn windowed_attention_gpu(
    q: &Tensor,           // [total_tokens, hidden]
    k: &Tensor,           // [total_tokens, hidden]
    v: &Tensor,           // [total_tokens, hidden]
    n_heads: usize,
    head_dim: usize,
    scale: f32,
    window_starts: &[usize],
    n_windows: usize,
) -> Result<Tensor> {
    let total_tokens = q.dim(0)?;
    let hidden = n_heads * head_dim;

    // Reshape to [total_tokens, n_heads, head_dim]
    let q = q.reshape(&[total_tokens, n_heads, head_dim])?;
    let k = k.reshape(&[total_tokens, n_heads, head_dim])?;
    let v = v.reshape(&[total_tokens, n_heads, head_dim])?;

    // If there's only 1 window covering everything, do full attention
    if n_windows == 1 {
        // [seq, heads, dim] -> [heads, seq, dim]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        // scores = Q @ K^T * scale -> [heads, seq, seq]
        let kt = k.transpose(1, 2)?.contiguous()?;
        let scores = q.matmul(&kt)?.affine(scale as f64, 0.0)?;
        let attn_weights = softmax_gpu(&scores)?;
        let out = attn_weights.matmul(&v)?; // [heads, seq, dim]
        let out = out.transpose(0, 1)?.contiguous()?.reshape(&[total_tokens, hidden])?;
        return Ok(out);
    }

    // Multiple windows: process each independently and concat
    let mut window_outputs = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let ws = window_starts[w];
        let we = window_starts[w + 1];
        let wlen = we - ws;
        if wlen == 0 { continue; }

        // Slice [ws..we, :, :]
        let qw = q.narrow(0, ws, wlen)?;
        let kw = k.narrow(0, ws, wlen)?;
        let vw = v.narrow(0, ws, wlen)?;

        // Transpose to [heads, wlen, dim]
        let qw = qw.transpose(0, 1)?.contiguous()?;
        let kw = kw.transpose(0, 1)?.contiguous()?;
        let vw = vw.transpose(0, 1)?.contiguous()?;

        let kwt = kw.transpose(1, 2)?.contiguous()?;
        let scores = qw.matmul(&kwt)?.affine(scale as f64, 0.0)?;
        let attn_weights = softmax_gpu(&scores)?;
        let out = attn_weights.matmul(&vw)?; // [heads, wlen, dim]
        let out = out.transpose(0, 1)?.contiguous()?; // [wlen, heads, dim]
        window_outputs.push(out);
    }

    // Cat along seq dimension -> [total_tokens, heads, dim] -> [total_tokens, hidden]
    let refs: Vec<&Tensor> = window_outputs.iter().collect();
    let combined = Tensor::cat(&refs, 0)?;
    combined.reshape(&[total_tokens, hidden])
}

// ---------------------------------------------------------------------------
// Encoder forward (Metal GPU)
// ---------------------------------------------------------------------------

/// Run the encoder forward pass with GPU acceleration.
///
/// Conv2D stem runs on CPU (small, constant-size per chunk).
/// Transformer layers run entirely on GPU — linear, layer-norm (Metal native
/// kernel), GELU, and windowed bidirectional attention (including softmax via
/// Metal native kernel). Only input upload and output download cross the
/// GPU↔CPU boundary.
pub fn encoder_forward_metal(
    weights: &EncoderGpuWeights,
    cfg: &QwenConfig,
    mel: &[f32],
    mel_frames: usize,
) -> Option<(Vec<f32>, usize)> {
    match encoder_forward_metal_inner(weights, cfg, mel, mel_frames) {
        Ok(result) => Some(result),
        Err(e) => {
            eprintln!("[metal] encoder_forward_metal failed: {}", e);
            None
        }
    }
}

fn encoder_forward_metal_inner(
    weights: &EncoderGpuWeights,
    cfg: &QwenConfig,
    mel: &[f32],
    mel_frames: usize,
) -> Result<(Vec<f32>, usize)> {
    let d_model = cfg.enc_d_model;
    let n_heads = cfg.enc_heads;
    let head_dim = cfg.enc_head_dim;
    let chunk_size = cfg.enc_chunk_size;
    let n_window_infer = cfg.enc_n_window_infer;

    // Tokens per full chunk (after 3× stride-2 conv)
    let tokens_per_chunk = {
        let w = chunk_size;
        let w1 = (w + 2 - 3) / 2 + 1;
        let w2 = (w1 + 2 - 3) / 2 + 1;
        (w2 + 2 - 3) / 2 + 1
    };

    let n_chunks = (mel_frames + chunk_size - 1) / chunk_size;

    // Pre-calculate per-chunk token counts
    let mut total_tokens = 0;
    let mut chunk_sizes = Vec::new();
    for c in 0..n_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(mel_frames);
        let chunk_w = end - start;
        let w1 = (chunk_w + 2 - 3) / 2 + 1;
        let w2 = (w1 + 2 - 3) / 2 + 1;
        let w3 = (w2 + 2 - 3) / 2 + 1;
        total_tokens += w3;
        chunk_sizes.push((start, end, w3));
    }

    // -----------------------------------------------------------------------
    // Conv2D stem on CPU (small, constant-size per chunk)
    // -----------------------------------------------------------------------
    let mut x_cpu = vec![0.0f32; total_tokens * d_model];
    let mut token_offset = 0;

    for &(start, end, w3) in &chunk_sizes {
        let chunk_w = end - start;

        // Extract chunk mel: [128, chunk_w]
        let mut chunk_mel = vec![0.0f32; 128 * chunk_w];
        for m in 0..128 {
            chunk_mel[m * chunk_w..(m + 1) * chunk_w]
                .copy_from_slice(&mel[m * mel_frames + start..m * mel_frames + end]);
        }

        // Conv1: [1, 128, chunk_w] -> [480, h1, w1]
        let h1 = (128 + 2 - 3) / 2 + 1;
        let w1 = (chunk_w + 2 - 3) / 2 + 1;
        let mut c1 = vec![0.0f32; CONV_HIDDEN * h1 * w1];
        kernels::conv2d(&mut c1, &chunk_mel, &weights.conv1_weight, Some(&weights.conv1_bias),
                        1, CONV_HIDDEN, 128, chunk_w, 3, 3, 2, 1);
        kernels::gelu(&mut c1, CONV_HIDDEN * h1 * w1);

        // Conv2: [480, h1, w1] -> [480, h2, w2]
        let h2 = (h1 + 2 - 3) / 2 + 1;
        let w2 = (w1 + 2 - 3) / 2 + 1;
        let mut c2 = vec![0.0f32; CONV_HIDDEN * h2 * w2];
        kernels::conv2d(&mut c2, &c1, &weights.conv2_weight, Some(&weights.conv2_bias),
                        CONV_HIDDEN, CONV_HIDDEN, h1, w1, 3, 3, 2, 1);
        kernels::gelu(&mut c2, CONV_HIDDEN * h2 * w2);

        // Conv3: [480, h2, w2] -> [480, h3, w3]
        let h3 = (h2 + 2 - 3) / 2 + 1;
        let mut c3 = vec![0.0f32; CONV_HIDDEN * h3 * w3];
        kernels::conv2d(&mut c3, &c2, &weights.conv3_weight, Some(&weights.conv3_bias),
                        CONV_HIDDEN, CONV_HIDDEN, h2, w2, 3, 3, 2, 1);
        kernels::gelu(&mut c3, CONV_HIDDEN * h3 * w3);

        // Reshape [480, h3, w3] -> [w3, 480*h3]
        let cpd_actual = CONV_HIDDEN * h3;
        let mut reshaped = vec![0.0f32; w3 * cpd_actual];
        for ch in 0..CONV_HIDDEN {
            for f in 0..h3 {
                let src_off = ch * h3 * w3 + f * w3;
                let dst_col = ch * h3 + f;
                for t in 0..w3 {
                    reshaped[t * cpd_actual + dst_col] = c3[src_off + t];
                }
            }
        }

        // Project [w3, conv_proj_dim] -> [w3, d_model] on GPU
        let reshaped_t = Tensor::from_slice(&reshaped, &[w3, cpd_actual], &Device::Cpu)
            ?.to_device(&weights.device)?;
        let projected = linear_nobias(&reshaped_t, &weights.conv_out_weight)?;

        // Add sinusoidal PE (computed on CPU, then added on GPU)
        let mut pe = vec![0.0f32; w3 * d_model];
        kernels::sinusoidal_pe(&mut pe, w3, d_model);
        let pe_t = Tensor::from_slice(&pe, &[w3, d_model], &Device::Cpu)
            ?.to_device(&weights.device)?;
        let with_pe = projected.add(&pe_t)?;

        // Copy back to CPU staging buffer
        let chunk_out = with_pe.to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        for (i, row) in chunk_out.iter().enumerate() {
            let dst = (token_offset + i) * d_model;
            x_cpu[dst..dst + d_model].copy_from_slice(row);
        }
        token_offset += w3;
    }

    // -----------------------------------------------------------------------
    // Upload full sequence to GPU for transformer layers
    // -----------------------------------------------------------------------
    let dev = &weights.device;
    let mut x = Tensor::from_slice(&x_cpu, &[total_tokens, d_model], &Device::Cpu)
        ?.to_device(dev)?;

    // Build attention window boundaries
    let window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
    let n_windows = (total_tokens + window_token_size - 1) / window_token_size;
    let mut window_starts_usize = vec![0usize; n_windows + 1];
    for w in 0..n_windows {
        window_starts_usize[w] = w * window_token_size;
    }
    window_starts_usize[n_windows] = total_tokens;

    let scale = 1.0 / (head_dim as f32).sqrt();

    // -----------------------------------------------------------------------
    // Transformer layers
    // -----------------------------------------------------------------------
    for layer in &weights.layers {
        // --- Self-attention ---

        // LayerNorm on CPU (fast, avoids Metal broadcast compatibility issues)
        let x_norm = layer_norm_gpu(&x, &layer.attn_norm_w, &layer.attn_norm_b, 1e-5)?;

        // Q, K, V projections on GPU
        let q_gpu = linear(&x_norm, &layer.wq, &layer.wq_bias)?;
        let k_gpu = linear(&x_norm, &layer.wk, &layer.wk_bias)?;
        let v_gpu = linear(&x_norm, &layer.wv, &layer.wv_bias)?;

        // Windowed bidirectional attention entirely on GPU
        let attn_out = windowed_attention_gpu(
            &q_gpu, &k_gpu, &v_gpu,
            n_heads, head_dim, scale,
            &window_starts_usize, n_windows,
        )?;

        // Output projection on GPU + residual
        let proj = linear(&attn_out, &layer.wo, &layer.wo_bias)?;
        x = x.add(&proj)?;

        // --- FFN ---
        let x_norm2 = layer_norm_gpu(&x, &layer.ffn_norm_w, &layer.ffn_norm_b, 1e-5)?;
        let ffn_up = linear(&x_norm2, &layer.fc1, &layer.fc1_bias)?;
        let ffn_act = ffn_up.gelu_erf()?;
        let ffn_down = linear(&ffn_act, &layer.fc2, &layer.fc2_bias)?;
        x = x.add(&ffn_down)?;
    }

    // -----------------------------------------------------------------------
    // Final post-processing
    // -----------------------------------------------------------------------
    let x_normed = layer_norm_gpu(&x, &weights.ln_post_w, &weights.ln_post_b, 1e-5)?;
    let proj1 = linear(&x_normed, &weights.proj1_w, &weights.proj1_b)?;
    let proj1_act = proj1.gelu_erf()?;
    let enc_output = linear(&proj1_act, &weights.proj2_w, &weights.proj2_b)?;

    // Copy final output back to CPU
    let output_cpu = enc_output.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?.to_vec1::<f32>()?;

    Ok((output_cpu, total_tokens))
}
