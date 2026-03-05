//! Top-level engine state ([`QwenCtx`]) owning all loaded weights and runtime buffers.

use crate::config::*;
use crate::decoder::*;
use crate::encoder::*;
use crate::encoder::EncoderBuffers;
use crate::kernels;
use crate::quantize::QuantFile;
use crate::safetensors::MultiSafetensors;
use crate::tokenizer::QwenTokenizer;

pub type TokenCallback = Box<dyn Fn(&str) + Send>;

/// Wrapper enum for BF16 or INT8 decoder.
pub enum DecoderKind {
    BF16(Decoder),
    Int8(DecoderInt8),
}

/// Wrapper enum for BF16 or INT8 encoder.
pub enum EncoderKind {
    BF16(Encoder),
    Int8(EncoderInt8),
}

/// Top-level ASR engine state owning model weights, KV cache, and scratch buffers.
///
/// Create with [`QwenCtx::load`], then pass to functions in the [`crate::transcribe`] module.
///
/// # Configurable fields
///
/// | Field | Default | Description |
/// |-------|---------|-------------|
/// | `segment_sec` | 0.0 | Segment duration for long audio (0 = no splitting) |
/// | `skip_silence` | false | Drop silent spans before transcription |
/// | `token_cb` | None | Streaming callback invoked for each decoded token |
/// | `prompt` | None | Optional text prompt (set via [`QwenCtx::set_prompt`]) |
/// | `force_language` | None | Force a language (set via [`QwenCtx::set_force_language`]) |
pub struct QwenCtx {
    pub config: QwenConfig,
    pub encoder: EncoderKind,
    pub decoder: DecoderKind,
    pub _safetensors: Option<MultiSafetensors>, // kept alive for mmap'd BF16 pointers
    pub _qint8: Option<QuantFile>,              // kept alive for mmap'd INT8/BF16 pointers
    pub model_dir: String,

    // KV cache
    pub kv_cache: KvCache,
    pub kv_initial_max_seq: usize,

    // Decoder buffers
    pub dec_bufs: DecoderBuffers,

    // Encoder scratch buffers (reusable across calls)
    pub enc_bufs: EncoderBuffers,

    // RoPE cache
    pub rope_cache: RopeCache,

    // Token streaming callback
    pub token_cb: Option<TokenCallback>,

    // Segmentation settings
    pub segment_sec: f32,
    pub search_sec: f32,

    // Streaming settings
    pub stream_chunk_sec: f32,
    pub stream_rollback: i32,
    pub stream_unfixed_chunks: i32,
    pub stream_max_new_tokens: i32,
    pub past_text_conditioning: bool,
    pub skip_silence: bool,
    pub use_gpu: bool,

    // Optional prompt/language
    pub prompt: Option<String>,
    pub force_language: Option<String>,
    pub prompt_tokens: Option<Vec<i32>>,
    pub force_prompt_tokens: Option<Vec<i32>>,
    pub prompt_tokens_ready: bool,

    // Perf stats
    pub perf_total_ms: f64,
    pub perf_text_tokens: i32,
    pub perf_audio_ms: f64,
    pub perf_encode_ms: f64,
    pub perf_decode_ms: f64,

    // GPU state (Metal acceleration)
    #[cfg(feature = "metal")]
    pub gpu_device: crate::device::ComputeDevice,
    #[cfg(feature = "metal")]
    pub gpu_encoder: Option<crate::encoder_gpu::EncoderGpuWeights>,
    #[cfg(feature = "metal")]
    pub gpu_decoder: Option<crate::decoder_gpu::DecoderGpuWeights>,
    #[cfg(feature = "metal")]
    pub gpu_kv_cache: Option<crate::decoder_gpu::GpuKvCache>,
    #[cfg(feature = "metal")]
    pub gpu_rope_cache: Option<crate::decoder_gpu::GpuRopeCache>,
    #[cfg(feature = "metal")]
    pub raw_decode: Option<crate::decoder_gpu::RawDecodeContext>,
}

impl QwenCtx {
    /// Get the BF16 token embeddings pointer (BF16 decoder only, panics on INT8).
    pub fn tok_embeddings_bf16(&self) -> *const u16 {
        match &self.decoder {
            DecoderKind::BF16(d) => d.tok_embeddings_bf16,
            DecoderKind::Int8(_) => panic!("tok_embeddings_bf16 not available on INT8 decoder"),
        }
    }

    /// Dequantize a single token embedding to f32 (works for both BF16 and INT8).
    pub fn tok_embed_to_f32(&self, dst: &mut [f32], token_id: i32, dim: usize) {
        match &self.decoder {
            DecoderKind::BF16(d) => {
                tok_embed_bf16_to_f32(dst, d.tok_embeddings_bf16, token_id, dim);
            }
            DecoderKind::Int8(d) => {
                tok_embed_int8_to_f32(dst, &d.tok_embeddings, token_id, dim);
            }
        }
    }

    /// Run encoder forward pass (dispatches to Metal GPU or CPU).
    pub fn encoder_forward(&mut self, mel: &[f32], mel_frames: usize) -> Option<(Vec<f32>, usize)> {
        #[cfg(feature = "metal")]
        {
            if self.use_gpu {
                if let Some(ref gpu_enc) = self.gpu_encoder {
                    return crate::encoder_gpu::encoder_forward_metal(gpu_enc, &self.config, mel, mel_frames);
                }
            }
        }
        let cfg = &self.config;
        match &self.encoder {
            EncoderKind::BF16(enc) => enc.forward(cfg, mel, mel_frames, Some(&mut self.enc_bufs)),
            EncoderKind::Int8(enc) => enc.forward(cfg, mel, mel_frames, Some(&mut self.enc_bufs)),
        }
    }

    /// Dispatch decoder prefill to BF16 or INT8 path (GPU-accelerated if available).
    pub fn decoder_prefill(&mut self, input_embeds: &[f32], seq_len: usize) {
        let cfg = &self.config;
        #[cfg(feature = "metal")]
        if self.use_gpu {
            // Phase 4: Full GPU prefill (no per-layer round-trips)
            if let (Some(ref gpu_dec), Some(ref mut gpu_kv), Some(ref mut gpu_rope)) =
                (&self.gpu_decoder, &mut self.gpu_kv_cache, &mut self.gpu_rope_cache)
            {
                crate::decoder_gpu::decoder_prefill_full_gpu(
                    gpu_dec, cfg, gpu_kv, gpu_rope,
                    &mut self.kv_cache, &mut self.rope_cache,
                    &mut self.dec_bufs, input_embeds, seq_len);
                return;
            }
            // Fallback: partial GPU prefill (Phase 3)
            if let Some(ref gpu_dec) = self.gpu_decoder {
                match &self.decoder {
                    DecoderKind::BF16(d) => {
                        crate::decoder_gpu::decoder_prefill_gpu(
                            gpu_dec, d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                            &mut self.dec_bufs, input_embeds, seq_len);
                        return;
                    }
                    DecoderKind::Int8(d) => {
                        crate::decoder_gpu::decoder_prefill_int8_gpu(
                            gpu_dec, d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                            &mut self.dec_bufs, input_embeds, seq_len);
                        return;
                    }
                }
            }
        }
        match &self.decoder {
            DecoderKind::BF16(d) => {
                decoder_prefill(d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                               &mut self.dec_bufs, input_embeds, seq_len);
            }
            DecoderKind::Int8(d) => {
                decoder_prefill_int8(d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                                    &mut self.dec_bufs, input_embeds, seq_len);
            }
        }
    }

    /// Dispatch single-token decoder forward to BF16 or INT8 path.
    pub fn decoder_forward(&mut self, input_embed: &[f32]) -> i32 {
        let cfg = &self.config;
        match &self.decoder {
            DecoderKind::BF16(d) => {
                decoder_forward(d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                               &mut self.dec_bufs, input_embed)
            }
            DecoderKind::Int8(d) => {
                decoder_forward_int8(d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                                    &mut self.dec_bufs, input_embed)
            }
        }
    }

    /// GPU-accelerated single-token forward from an embedding slice.
    ///
    /// Runs transformer layers on GPU, then final norm + LM head argmax on CPU.
    /// Returns `None` if GPU is not available or forward fails.
    #[cfg(feature = "metal")]
    pub fn decoder_forward_gpu_embed(&mut self, input_embed: &[f32]) -> Option<i32> {
        // Phase 6: Raw Metal decode (bypasses candle Tensor API)
        if let (Some(ref raw), Some(ref gpu_dec), Some(ref mut gpu_kv), Some(ref mut gpu_rope)) =
            (&self.raw_decode, &self.gpu_decoder, &mut self.gpu_kv_cache, &mut self.gpu_rope_cache)
        {
            match crate::decoder_gpu::decoder_forward_raw(
                raw, gpu_dec, &self.config, gpu_kv, gpu_rope, &mut self.rope_cache, input_embed
            ) {
                Ok(hidden) => {
                    let gpu_kv_len = gpu_kv.len;

                    // Final RMS norm on CPU
                    let dim = self.config.dec_hidden;
                    let eps = self.config.dec_rms_norm_eps;
                    let norm_weights = match &self.decoder {
                        DecoderKind::BF16(d) => &d.norm,
                        DecoderKind::Int8(d) => &d.norm,
                    };
                    let mut x_normed = vec![0.0f32; dim];
                    kernels::rms_norm(&mut x_normed, &hidden, norm_weights, 1, dim, eps);

                    // LM head argmax on CPU
                    let lm_out_dim = self.config.lm_head_dim();
                    let next_token = match &self.decoder {
                        DecoderKind::BF16(d) => {
                            let lm_weight = d.lm_head_bf16.unwrap_or(d.tok_embeddings_bf16);
                            kernels::argmax_matvec_bf16(&x_normed, lm_weight, dim, lm_out_dim) as i32
                        }
                        DecoderKind::Int8(d) => {
                            let lm = d.lm_head.as_ref().unwrap_or(&d.tok_embeddings);
                            let mut logits = vec![0.0f32; lm_out_dim];
                            kernels::linear_nobias_int8(&mut logits, &x_normed,
                                                        lm.data, lm.scales, 1, dim, lm_out_dim);
                            logits.iter().enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(i, _)| i as i32)
                                .unwrap_or(0)
                        }
                    };

                    // Keep CPU KV cache len in sync
                    self.kv_cache.len = gpu_kv_len;
                    return Some(next_token);
                }
                Err(e) => {
                    eprintln!("[metal] raw GPU decode failed: {}, falling back", e);
                }
            }
        }

        // Tensor-based GPU decode
        if let (Some(ref gpu_dec), Some(ref mut gpu_kv), Some(ref mut gpu_rope)) =
            (&self.gpu_decoder, &mut self.gpu_kv_cache, &mut self.gpu_rope_cache)
        {
            match crate::decoder_gpu::decoder_forward_gpu(
                gpu_dec, &self.config, gpu_kv, gpu_rope, &mut self.rope_cache, input_embed
            ) {
                Ok(hidden) => {
                    let gpu_kv_len = gpu_kv.len;

                    // Final RMS norm on CPU
                    let dim = self.config.dec_hidden;
                    let eps = self.config.dec_rms_norm_eps;
                    let norm_weights = match &self.decoder {
                        DecoderKind::BF16(d) => &d.norm,
                        DecoderKind::Int8(d) => &d.norm,
                    };
                    let mut x_normed = vec![0.0f32; dim];
                    kernels::rms_norm(&mut x_normed, &hidden, norm_weights, 1, dim, eps);

                    // LM head argmax on CPU
                    let lm_out_dim = self.config.lm_head_dim();
                    let next_token = match &self.decoder {
                        DecoderKind::BF16(d) => {
                            let lm_weight = d.lm_head_bf16.unwrap_or(d.tok_embeddings_bf16);
                            kernels::argmax_matvec_bf16(&x_normed, lm_weight, dim, lm_out_dim) as i32
                        }
                        DecoderKind::Int8(d) => {
                            let lm = d.lm_head.as_ref().unwrap_or(&d.tok_embeddings);
                            let mut logits = vec![0.0f32; lm_out_dim];
                            kernels::linear_nobias_int8(&mut logits, &x_normed,
                                                        lm.data, lm.scales, 1, dim, lm_out_dim);
                            logits.iter().enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(i, _)| i as i32)
                                .unwrap_or(0)
                        }
                    };

                    // Keep CPU KV cache len in sync
                    self.kv_cache.len = gpu_kv_len;
                    return Some(next_token);
                }
                Err(e) => {
                    eprintln!("[metal] GPU decode failed: {}, falling back to CPU", e);
                }
            }
        }
        None
    }

    /// Full single-token decode: token_id in → next token_id out.
    /// Uses CPU for autoregressive decode (GPU launch overhead too high for single-token).
    pub fn decoder_forward_token(&mut self, token_id: i32) -> i32 {
        let dim = self.config.dec_hidden;
        let mut tmp_embed = vec![0.0f32; dim];
        self.tok_embed_to_f32(&mut tmp_embed, token_id, dim);
        self.decoder_forward(&tmp_embed)
    }

    /// Dispatch decoder prefill with logits (for forced aligner).
    pub fn decoder_prefill_logits(&mut self, input_embeds: &[f32], seq_len: usize) -> Vec<f32> {
        let cfg = &self.config;
        match &self.decoder {
            DecoderKind::BF16(d) => {
                decoder_prefill_logits(d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                                      &mut self.dec_bufs, input_embeds, seq_len)
            }
            DecoderKind::Int8(d) => {
                decoder_prefill_logits_int8(d, cfg, &mut self.kv_cache, &mut self.rope_cache,
                                           &mut self.dec_bufs, input_embeds, seq_len)
            }
        }
    }

    /// Load a Qwen3-ASR model from `model_dir`.
    ///
    /// Supports three scenarios:
    /// 1. V2 self-contained qint8 (no safetensors needed)
    /// 2. V1 qint8 + safetensors (legacy)
    /// 3. Pure BF16 safetensors (no qint8)
    ///
    /// Returns `None` if any required file is missing or malformed.
    pub fn load(model_dir: &str) -> Option<Self> {
        if kernels::verbose() >= 1 {
            eprintln!("Loading model from {}", model_dir);
        }

        // Try to open qint8 file
        let qint8_path = format!("{}/model_int8.qint8", model_dir);
        let qf = QuantFile::open(&qint8_path);

        let is_v2 = qf.as_ref().map_or(false, |q| q.is_self_contained());

        if is_v2 {
            // ---- V2 self-contained: everything from qint8, no safetensors ----
            let qf = qf.unwrap();
            if kernels::verbose() >= 1 {
                eprintln!("Found V2 self-contained qint8 (no safetensors needed)");
            }

            // Detect config from qint8 tensor metadata
            // Convert usize shapes to i64 for DetectInfo compatibility
            let lm_head_shape_i64: Option<Vec<i64>> = qf.find("thinker.lm_head.weight")
                .map(|t| t.shape.iter().map(|&s| s as i64).collect());
            let embed_shape_i64: Option<Vec<i64>> = qf.find("thinker.model.embed_tokens.weight")
                .map(|t| t.shape.iter().map(|&s| s as i64).collect());
            let gate_shape_i64: Option<Vec<i64>> = qf.find("thinker.model.layers.0.mlp.gate_proj.weight")
                .map(|t| t.shape.iter().map(|&s| s as i64).collect());
            let info = crate::config::DetectInfo {
                has_enc_layer_18: qf.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight"),
                lm_head_shape: lm_head_shape_i64.as_deref(),
                embed_tokens_shape: embed_shape_i64.as_deref(),
                gate_proj_shape: gate_shape_i64.as_deref(),
            };
            let cfg = QwenConfig::detect(&info);

            if kernels::verbose() >= 1 {
                let variant = if cfg.dec_hidden >= 2048 { "1.7B" } else { "0.6B" };
                let model_type = if cfg.is_aligner() { "ForcedAligner" } else { "ASR" };
                eprintln!("Detected: Qwen3-{}-{} (INT8)", model_type, variant);
            }

            // Load INT8 encoder from qint8
            if kernels::verbose() >= 1 {
                eprintln!("Loading INT8 encoder weights from qint8...");
            }
            let encoder = EncoderInt8::load_from_qint8(&qf, &cfg)?;

            // Load INT8 decoder from qint8
            if kernels::verbose() >= 1 {
                eprintln!("Loading INT8 decoder weights from qint8...");
            }
            let decoder_int8 = DecoderInt8::load(&qf, &cfg)?;

            let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
            let kv_cache = KvCache::new(cfg.dec_layers, 256, kv_dim);
            let dec_bufs = DecoderBuffers::new(&cfg);

            if kernels::verbose() >= 1 {
                eprintln!("Model loaded (INT8 self-contained).");
            }

            #[cfg(feature = "metal")]
            let gpu_device = crate::device::ComputeDevice::best();

            #[cfg(feature = "metal")]
            let gpu_encoder = {
                if let crate::device::ComputeDevice::Metal(ref dev) = gpu_device {
                    if kernels::verbose() >= 1 {
                        eprintln!("[metal] Uploading encoder weights to GPU...");
                    }
                    match crate::encoder_gpu::EncoderGpuWeights::from_encoder_int8(&encoder, &cfg, dev) {
                        Ok(w) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] Encoder weights uploaded.");
                            }
                            Some(w)
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to upload encoder weights: {}, falling back to CPU", e);
                            None
                        }
                    }
                } else {
                    None
                }
            };

            #[cfg(feature = "metal")]
            let gpu_decoder = {
                if let crate::device::ComputeDevice::Metal(ref dev) = gpu_device {
                    if kernels::verbose() >= 1 {
                        eprintln!("[metal] Uploading decoder weights to GPU...");
                    }
                    match crate::decoder_gpu::DecoderGpuWeights::from_decoder_int8(&decoder_int8, &cfg, dev) {
                        Ok(w) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] Decoder weights uploaded.");
                            }
                            Some(w)
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to upload decoder weights: {}, falling back to CPU", e);
                            None
                        }
                    }
                } else {
                    None
                }
            };

            #[cfg(feature = "metal")]
            let (gpu_kv_cache, gpu_rope_cache) = {
                if let (Some(_), crate::device::ComputeDevice::Metal(ref dev)) = (&gpu_decoder, &gpu_device) {
                    match crate::decoder_gpu::GpuKvCache::new(
                        cfg.dec_layers, cfg.dec_kv_heads, cfg.dec_head_dim, 2048, dev
                    ) {
                        Ok(kv) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] GPU KV cache + RoPE cache initialized.");
                            }
                            let rope = crate::decoder_gpu::GpuRopeCache::new(dev, cfg.dec_head_dim);
                            (Some(kv), Some(rope))
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to create GPU KV cache: {}", e);
                            (None, None)
                        }
                    }
                } else {
                    (None, None)
                }
            };

            #[cfg(feature = "metal")]
            let raw_decode = {
                if let Some(ref gpu_dec) = gpu_decoder {
                    match crate::decoder_gpu::RawDecodeContext::new(gpu_dec, &cfg) {
                        Ok(r) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] Raw decode context initialized (Phase 6).");
                            }
                            Some(r)
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to create raw decode context: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            };

            Some(QwenCtx {
                config: cfg,
                encoder: EncoderKind::Int8(encoder),
                decoder: DecoderKind::Int8(decoder_int8),
                _safetensors: None,
                _qint8: Some(qf),
                model_dir: model_dir.to_string(),
                kv_cache,
                kv_initial_max_seq: 256,
                dec_bufs,
                enc_bufs: EncoderBuffers::new(),
                rope_cache: RopeCache::new(),
                token_cb: None,
                segment_sec: 0.0,
                search_sec: 3.0,
                stream_chunk_sec: 2.0,
                stream_rollback: 5,
                stream_unfixed_chunks: 2,
                stream_max_new_tokens: 32,
                past_text_conditioning: false,
                skip_silence: false,
                use_gpu: true,
                prompt: None,
                force_language: None,
                prompt_tokens: None,
                force_prompt_tokens: None,
                prompt_tokens_ready: false,
                perf_total_ms: 0.0,
                perf_text_tokens: 0,
                perf_audio_ms: 0.0,
                perf_encode_ms: 0.0,
                perf_decode_ms: 0.0,
                #[cfg(feature = "metal")]
                gpu_device,
                #[cfg(feature = "metal")]
                gpu_encoder,
                #[cfg(feature = "metal")]
                gpu_decoder,
                #[cfg(feature = "metal")]
                gpu_kv_cache,
                #[cfg(feature = "metal")]
                gpu_rope_cache,
                #[cfg(feature = "metal")]
                raw_decode,
            })
        } else {
            // ---- BF16 path: requires safetensors ----
            let ms = MultiSafetensors::open(model_dir)?;

            let info = crate::config::DetectInfo {
                has_enc_layer_18: ms.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight"),
                lm_head_shape: ms.find("thinker.lm_head.weight").map(|(_, t)| t.shape.as_slice()),
                embed_tokens_shape: ms.find("thinker.model.embed_tokens.weight").map(|(_, t)| t.shape.as_slice()),
                gate_proj_shape: ms.find("thinker.model.layers.0.mlp.gate_proj.weight").map(|(_, t)| t.shape.as_slice()),
            };
            let cfg = QwenConfig::detect(&info);

            if kernels::verbose() >= 1 {
                let variant = if cfg.dec_hidden >= 2048 { "1.7B" } else { "0.6B" };
                let model_type = if cfg.is_aligner() { "ForcedAligner" } else { "ASR" };
                eprintln!("Detected: Qwen3-{}-{}", model_type, variant);
                if cfg.is_aligner() {
                    eprintln!("  classify_num={}, timestamp_segment_time={:.0}ms",
                              cfg.classify_num, cfg.timestamp_segment_time);
                    eprintln!("  encoder: {}d {}L, decoder: {}d {}L",
                              cfg.enc_d_model, cfg.enc_layers, cfg.dec_hidden, cfg.dec_layers);
                }
            }

            if kernels::verbose() >= 1 {
                eprintln!("Loading encoder weights...");
            }
            let encoder = Encoder::load(&ms, &cfg)?;

            if kernels::verbose() >= 1 {
                eprintln!("Loading decoder weights...");
            }
            let decoder = Decoder::load(&ms, &cfg)?;

            let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
            let kv_cache = KvCache::new(cfg.dec_layers, 256, kv_dim);
            let dec_bufs = DecoderBuffers::new(&cfg);

            if kernels::verbose() >= 1 {
                eprintln!("Model loaded.");
            }

            #[cfg(feature = "metal")]
            let gpu_device = crate::device::ComputeDevice::best();

            #[cfg(feature = "metal")]
            let gpu_encoder = {
                if let crate::device::ComputeDevice::Metal(ref dev) = gpu_device {
                    if kernels::verbose() >= 1 {
                        eprintln!("[metal] Uploading encoder weights to GPU...");
                    }
                    match crate::encoder_gpu::EncoderGpuWeights::from_encoder(&encoder, &cfg, dev) {
                        Ok(w) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] Encoder weights uploaded.");
                            }
                            Some(w)
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to upload encoder weights: {}, falling back to CPU", e);
                            None
                        }
                    }
                } else {
                    None
                }
            };

            #[cfg(feature = "metal")]
            let gpu_decoder = {
                if let crate::device::ComputeDevice::Metal(ref dev) = gpu_device {
                    if kernels::verbose() >= 1 {
                        eprintln!("[metal] Uploading decoder weights to GPU...");
                    }
                    match crate::decoder_gpu::DecoderGpuWeights::from_decoder(&decoder, &cfg, dev) {
                        Ok(w) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] Decoder weights uploaded.");
                            }
                            Some(w)
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to upload decoder weights: {}, falling back to CPU", e);
                            None
                        }
                    }
                } else {
                    None
                }
            };

            #[cfg(feature = "metal")]
            let (gpu_kv_cache, gpu_rope_cache) = {
                if let (Some(_), crate::device::ComputeDevice::Metal(ref dev)) = (&gpu_decoder, &gpu_device) {
                    match crate::decoder_gpu::GpuKvCache::new(
                        cfg.dec_layers, cfg.dec_kv_heads, cfg.dec_head_dim, 2048, dev
                    ) {
                        Ok(kv) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] GPU KV cache + RoPE cache initialized.");
                            }
                            let rope = crate::decoder_gpu::GpuRopeCache::new(dev, cfg.dec_head_dim);
                            (Some(kv), Some(rope))
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to create GPU KV cache: {}", e);
                            (None, None)
                        }
                    }
                } else {
                    (None, None)
                }
            };

            #[cfg(feature = "metal")]
            let raw_decode = {
                if let Some(ref gpu_dec) = gpu_decoder {
                    match crate::decoder_gpu::RawDecodeContext::new(gpu_dec, &cfg) {
                        Ok(r) => {
                            if kernels::verbose() >= 1 {
                                eprintln!("[metal] Raw decode context initialized (Phase 6).");
                            }
                            Some(r)
                        }
                        Err(e) => {
                            eprintln!("[metal] Failed to create raw decode context: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            };

            Some(QwenCtx {
                config: cfg,
                encoder: EncoderKind::BF16(encoder),
                decoder: DecoderKind::BF16(decoder),
                _safetensors: Some(ms),
                _qint8: None,
                model_dir: model_dir.to_string(),
                kv_cache,
                kv_initial_max_seq: 256,
                dec_bufs,
                enc_bufs: EncoderBuffers::new(),
                rope_cache: RopeCache::new(),
                token_cb: None,
                segment_sec: 0.0,
                search_sec: 3.0,
                stream_chunk_sec: 2.0,
                stream_rollback: 5,
                stream_unfixed_chunks: 2,
                stream_max_new_tokens: 32,
                past_text_conditioning: false,
                skip_silence: false,
                use_gpu: true,
                prompt: None,
                force_language: None,
                prompt_tokens: None,
                force_prompt_tokens: None,
                prompt_tokens_ready: false,
                perf_total_ms: 0.0,
                perf_text_tokens: 0,
                perf_audio_ms: 0.0,
                perf_encode_ms: 0.0,
                perf_decode_ms: 0.0,
                #[cfg(feature = "metal")]
                gpu_device,
                #[cfg(feature = "metal")]
                gpu_encoder,
                #[cfg(feature = "metal")]
                gpu_decoder,
                #[cfg(feature = "metal")]
                gpu_kv_cache,
                #[cfg(feature = "metal")]
                gpu_rope_cache,
                #[cfg(feature = "metal")]
                raw_decode,
            })
        }
    }

    /// Set an optional text prompt to guide transcription. Pass an empty string to clear.
    pub fn set_prompt(&mut self, prompt: &str) -> Result<(), ()> {
        if prompt.is_empty() {
            self.prompt = None;
        } else {
            self.prompt = Some(prompt.to_string());
        }
        self.prompt_tokens_ready = false;
        Ok(())
    }

    /// Force a specific language (e.g. `"English"`, `"Chinese"`). Pass an empty
    /// string for auto-detection. Returns `Err(())` if the language is not recognized.
    pub fn set_force_language(&mut self, language: &str) -> Result<(), ()> {
        if language.is_empty() {
            self.force_language = None;
            self.prompt_tokens_ready = false;
            return Ok(());
        }

        match normalize_language(language) {
            Some(normalized) => {
                self.force_language = Some(normalized);
                self.prompt_tokens_ready = false;
                Ok(())
            }
            None => Err(()),
        }
    }

    pub fn prepare_prompt_tokens(&mut self, tokenizer: &QwenTokenizer) -> bool {
        if self.prompt_tokens_ready {
            return true;
        }

        self.prompt_tokens = None;
        self.force_prompt_tokens = None;

        if let Some(ref prompt) = self.prompt {
            match tokenizer.encode(prompt) {
                Some(tokens) => self.prompt_tokens = Some(tokens),
                None => {
                    eprintln!("qwen: failed to encode --prompt text");
                    return false;
                }
            }
        }

        if let Some(ref lang) = self.force_language {
            let force_text = format!("language {}", lang);
            match tokenizer.encode(&force_text) {
                Some(mut lang_tokens) => {
                    lang_tokens.push(TOKEN_ASR_TEXT);
                    self.force_prompt_tokens = Some(lang_tokens);
                }
                None => {
                    eprintln!("qwen: failed to encode --language text");
                    return false;
                }
            }
        }

        self.prompt_tokens_ready = true;
        true
    }

    pub fn reset_perf(&mut self) {
        self.perf_total_ms = 0.0;
        self.perf_text_tokens = 0;
        self.perf_audio_ms = 0.0;
        self.perf_encode_ms = 0.0;
        self.perf_decode_ms = 0.0;
    }
}
