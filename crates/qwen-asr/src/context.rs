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
    pub encoder: Encoder,
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
}

impl QwenCtx {
    /// Get the BF16 token embeddings pointer (works for both BF16 and INT8 decoder).
    pub fn tok_embeddings_bf16(&self) -> *const u16 {
        match &self.decoder {
            DecoderKind::BF16(d) => d.tok_embeddings_bf16,
            DecoderKind::Int8(d) => d.tok_embeddings_bf16,
        }
    }

    /// Dispatch decoder prefill to BF16 or INT8 path.
    pub fn decoder_prefill(&mut self, input_embeds: &[f32], seq_len: usize) {
        let cfg = &self.config;
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

            // Load encoder from qint8
            if kernels::verbose() >= 1 {
                eprintln!("Loading encoder weights from qint8...");
            }
            let encoder = Encoder::load_from_qint8(&qf, &cfg)?;

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

            Some(QwenCtx {
                config: cfg,
                encoder,
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

            Some(QwenCtx {
                config: cfg,
                encoder,
                decoder: DecoderKind::BF16(decoder),
                _safetensors: Some(ms),
                _qint8: qf, // may be V1 qint8, kept for future use
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
