//! Offline quantization tool: BF16 safetensors → INT8 `.qint8`
//!
//! Usage:
//!     qwen-asr-quantize <model_dir> [output_path]          # ASR model (thinker.* prefix)
//!     qwen-asr-quantize --llm <model_dir> [output_path]    # Standalone LLM (model.* prefix)
//!
//! Reads `model*.safetensors` from `model_dir`, quantizes weights to per-channel
//! symmetric INT8, and writes a V2 `.qint8` file.

use qwen_asr::config::QwenConfig;
use qwen_asr::quantize::{
    quantize_bf16_to_int8, write_qint8_v2_file,
    BF16WriteEntry, F32WriteEntry, QuantWriteEntry,
};
use qwen_asr::safetensors::MultiSafetensors;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --llm flag
    let is_llm = args.iter().any(|a| a == "--llm");
    let positional: Vec<&String> = args[1..].iter().filter(|a| *a != "--llm").collect();

    if positional.is_empty() {
        eprintln!("Usage: qwen-asr-quantize [--llm] <model_dir> [output_path]");
        eprintln!("  --llm    Quantize a standalone LLM (e.g. Qwen3-0.6B) instead of ASR model");
        std::process::exit(1);
    }

    let model_dir = positional[0].as_str();
    let output_path = if positional.len() >= 2 {
        positional[1].clone()
    } else {
        format!("{}/model_int8.qint8", model_dir)
    };

    eprintln!("Loading safetensors from {} ...", model_dir);
    let ms = MultiSafetensors::open(model_dir).unwrap_or_else(|| {
        eprintln!("Failed to open safetensors in {}", model_dir);
        std::process::exit(1);
    });

    if is_llm {
        quantize_llm(&ms, &output_path);
    } else {
        quantize_asr(&ms, &output_path);
    }
}

/// Quantize a standalone LLM (e.g. Qwen3-0.6B).
/// Reads weights with `model.*` prefix, writes with `thinker.model.*` prefix
/// so that DecoderInt8::load() can load them directly.
fn quantize_llm(ms: &MultiSafetensors, output_path: &str) {
    // Detect config from standalone LLM tensor names (no "thinker." prefix)
    let info = qwen_asr::config::DetectInfo {
        has_enc_layer_18: false,
        lm_head_shape: ms.find("lm_head.weight").map(|(_, t)| t.shape.as_slice()),
        embed_tokens_shape: ms.find("model.embed_tokens.weight").map(|(_, t)| t.shape.as_slice()),
        gate_proj_shape: ms
            .find("model.layers.0.mlp.gate_proj.weight")
            .map(|(_, t)| t.shape.as_slice()),
    };
    let cfg = QwenConfig::detect(&info);

    let variant = if cfg.dec_hidden >= 2048 { "1.7B" } else { "0.6B" };
    eprintln!("Detected: Qwen3-{} (standalone LLM)", variant);
    eprintln!(
        "  Decoder: {} layers, hidden={}, intermediate={}, heads={}, kv_heads={}",
        cfg.dec_layers, cfg.dec_hidden, cfg.dec_intermediate, cfg.dec_heads, cfg.dec_kv_heads
    );

    let mut quant_tensors: Vec<QuantWriteEntry> = Vec::new();
    let mut f32_tensors: Vec<F32WriteEntry> = Vec::new();

    // ---- Quantize decoder layer weights ----
    // Read from "model.layers.N.*", write as "thinker.model.layers.N.*"
    for i in 0..cfg.dec_layers {
        let src_lp = format!("model.layers.{}", i);
        let dst_lp = format!("thinker.model.layers.{}", i);
        eprint!("  Quantizing layer {}/{} ...\r", i + 1, cfg.dec_layers);

        let weight_suffixes = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];

        for suffix in &weight_suffixes {
            let src_name = format!("{}.{}", src_lp, suffix);
            let dst_name = format!("{}.{}", dst_lp, suffix);
            quantize_and_push_rename(ms, &src_name, &dst_name, &mut quant_tensors);
        }

        // Norm tensors: keep as f32 (1D, small)
        let norm_suffixes = [
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ];

        for suffix in &norm_suffixes {
            let src_name = format!("{}.{}", src_lp, suffix);
            let dst_name = format!("{}.{}", dst_lp, suffix);
            let data = ms.get_f32(&src_name).unwrap_or_else(|| {
                eprintln!("\nNorm weight not found: {}", src_name);
                std::process::exit(1);
            });
            let (_, tmeta) = ms.find(&src_name).unwrap();
            let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
            f32_tensors.push(F32WriteEntry {
                name: dst_name,
                shape,
                data,
            });
        }
    }
    eprintln!("  Quantized {} decoder layers.            ", cfg.dec_layers);

    // ---- Final norm ----
    {
        let src_name = "model.norm.weight";
        let dst_name = "thinker.model.norm.weight";
        let data = ms.get_f32(src_name).unwrap_or_else(|| {
            eprintln!("Norm weight not found: {}", src_name);
            std::process::exit(1);
        });
        let (_, tmeta) = ms.find(src_name).unwrap();
        let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
        f32_tensors.push(F32WriteEntry {
            name: dst_name.to_string(),
            shape,
            data,
        });
    }

    // ---- lm_head ----
    if let Some((_, tmeta)) = ms.find("lm_head.weight") {
        let out_dim = tmeta.shape[0] as usize;
        let in_dim = tmeta.shape[1] as usize;
        eprintln!("  Quantizing lm_head ({} x {}) ...", out_dim, in_dim);

        let bf16_ptr = ms.get_bf16_direct("lm_head.weight").unwrap();
        let (int8_data, scales) = quantize_bf16_to_int8(bf16_ptr, out_dim, in_dim);

        quant_tensors.push(QuantWriteEntry {
            name: "thinker.lm_head.weight".to_string(),
            shape: vec![out_dim, in_dim],
            int8_data,
            scales,
        });
    } else {
        eprintln!("  lm_head not found (likely tied with embeddings), skipping.");
    }

    // ---- Token embeddings (INT8) ----
    {
        eprintln!("  Quantizing token embeddings ...");
        quantize_and_push_rename(
            ms,
            "model.embed_tokens.weight",
            "thinker.model.embed_tokens.weight",
            &mut quant_tensors,
        );
    }

    // ---- Write V2 output (no encoder, no BF16 tensors) ----
    let bf16_tensors: Vec<BF16WriteEntry> = Vec::new();
    let n_quant = quant_tensors.len();
    let n_f32 = f32_tensors.len();
    eprintln!(
        "Writing {} INT8 + {} F32 tensors to {} ...",
        n_quant, n_f32, output_path
    );

    write_qint8_v2_file(output_path, &quant_tensors, &f32_tensors, &bf16_tensors)
        .unwrap_or_else(|e| {
            eprintln!("Failed to write output: {}", e);
            std::process::exit(1);
        });

    let output_size = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "Done! Output: {} ({:.1} MB)",
        output_path,
        output_size as f64 / 1024.0 / 1024.0
    );
}

/// Quantize an ASR model (original logic with thinker.* prefix + encoder).
fn quantize_asr(ms: &MultiSafetensors, output_path: &str) {
    let info = qwen_asr::config::DetectInfo {
        has_enc_layer_18: ms.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight"),
        lm_head_shape: ms.find("thinker.lm_head.weight").map(|(_, t)| t.shape.as_slice()),
        embed_tokens_shape: ms
            .find("thinker.model.embed_tokens.weight")
            .map(|(_, t)| t.shape.as_slice()),
        gate_proj_shape: ms
            .find("thinker.model.layers.0.mlp.gate_proj.weight")
            .map(|(_, t)| t.shape.as_slice()),
    };
    let cfg = QwenConfig::detect(&info);

    let variant = if cfg.dec_hidden >= 2048 { "1.7B" } else { "0.6B" };
    let model_type = if cfg.is_aligner() {
        "ForcedAligner"
    } else {
        "ASR"
    };
    eprintln!("Detected: Qwen3-{}-{}", model_type, variant);
    eprintln!(
        "  Decoder: {} layers, hidden={}, intermediate={}, heads={}, kv_heads={}",
        cfg.dec_layers, cfg.dec_hidden, cfg.dec_intermediate, cfg.dec_heads, cfg.dec_kv_heads
    );

    let mut quant_tensors: Vec<QuantWriteEntry> = Vec::new();
    let mut f32_tensors: Vec<F32WriteEntry> = Vec::new();

    // ---- Quantize decoder layer weights ----
    for i in 0..cfg.dec_layers {
        let lp = format!("thinker.model.layers.{}", i);
        eprint!("  Quantizing layer {}/{} ...\r", i + 1, cfg.dec_layers);

        let weight_names = [
            format!("{}.self_attn.q_proj.weight", lp),
            format!("{}.self_attn.k_proj.weight", lp),
            format!("{}.self_attn.v_proj.weight", lp),
            format!("{}.self_attn.o_proj.weight", lp),
            format!("{}.mlp.gate_proj.weight", lp),
            format!("{}.mlp.up_proj.weight", lp),
            format!("{}.mlp.down_proj.weight", lp),
        ];

        for wname in &weight_names {
            quantize_and_push(ms, wname, &mut quant_tensors);
        }

        let norm_names = [
            format!("{}.self_attn.q_norm.weight", lp),
            format!("{}.self_attn.k_norm.weight", lp),
            format!("{}.input_layernorm.weight", lp),
            format!("{}.post_attention_layernorm.weight", lp),
        ];

        for nname in &norm_names {
            let data = ms.get_f32(nname).unwrap_or_else(|| {
                eprintln!("\nNorm weight not found: {}", nname);
                std::process::exit(1);
            });
            let (_, tmeta) = ms.find(nname).unwrap();
            let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
            f32_tensors.push(F32WriteEntry {
                name: nname.clone(),
                shape,
                data,
            });
        }
    }

    eprintln!("  Quantized {} decoder layers.            ", cfg.dec_layers);

    // ---- Final norm ----
    {
        let name = "thinker.model.norm.weight";
        let data = ms.get_f32(name).unwrap_or_else(|| {
            eprintln!("Norm weight not found: {}", name);
            std::process::exit(1);
        });
        let (_, tmeta) = ms.find(name).unwrap();
        let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
        f32_tensors.push(F32WriteEntry {
            name: name.to_string(),
            shape,
            data,
        });
    }

    // ---- lm_head ----
    if let Some((_, tmeta)) = ms.find("thinker.lm_head.weight") {
        let out_dim = tmeta.shape[0] as usize;
        let in_dim = tmeta.shape[1] as usize;
        eprintln!("  Quantizing lm_head ({} x {}) ...", out_dim, in_dim);

        let bf16_ptr = ms.get_bf16_direct("thinker.lm_head.weight").unwrap();
        let (int8_data, scales) = quantize_bf16_to_int8(bf16_ptr, out_dim, in_dim);

        quant_tensors.push(QuantWriteEntry {
            name: "thinker.lm_head.weight".to_string(),
            shape: vec![out_dim, in_dim],
            int8_data,
            scales,
        });
    } else {
        eprintln!("  lm_head not found (likely tied with embeddings), skipping.");
    }

    // ---- V2: Pack encoder weights ----
    let bf16_tensors: Vec<BF16WriteEntry> = Vec::new();

    let enc_prefix = "thinker.audio_tower.";
    eprintln!("  Packing encoder weights (V2 self-contained) ...");

    for conv_name in &["conv2d1", "conv2d2", "conv2d3"] {
        for suffix in &["weight", "bias"] {
            let name = format!("{}{}.{}", enc_prefix, conv_name, suffix);
            let data = ms.get_f32(&name).unwrap_or_else(|| {
                eprintln!("Encoder weight not found: {}", name);
                std::process::exit(1);
            });
            let (_, tmeta) = ms.find(&name).unwrap();
            let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
            f32_tensors.push(F32WriteEntry { name, shape, data });
        }
    }

    {
        let name = format!("{}conv_out.weight", enc_prefix);
        quantize_and_push(ms, &name, &mut quant_tensors);
    }

    for i in 0..cfg.enc_layers {
        let lp = format!("{}layers.{}", enc_prefix, i);
        eprint!("  Quantizing encoder layer {}/{} ...\r", i + 1, cfg.enc_layers);

        for suffix in &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.out_proj.weight",
            "fc1.weight",
            "fc2.weight",
        ] {
            let name = format!("{}.{}", lp, suffix);
            quantize_and_push(ms, &name, &mut quant_tensors);
        }

        for suffix in &[
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "fc1.bias",
            "fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ] {
            let name = format!("{}.{}", lp, suffix);
            let data = ms.get_f32(&name).unwrap_or_else(|| {
                eprintln!("Encoder weight not found: {}", name);
                std::process::exit(1);
            });
            let (_, tmeta) = ms.find(&name).unwrap();
            let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
            f32_tensors.push(F32WriteEntry { name, shape, data });
        }
    }
    eprintln!("  Quantized {} encoder layers.                ", cfg.enc_layers);

    for suffix in &["ln_post.weight", "ln_post.bias", "proj1.bias", "proj2.bias"] {
        let name = format!("{}{}", enc_prefix, suffix);
        let data = ms.get_f32(&name).unwrap_or_else(|| {
            eprintln!("Encoder weight not found: {}", name);
            std::process::exit(1);
        });
        let (_, tmeta) = ms.find(&name).unwrap();
        let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
        f32_tensors.push(F32WriteEntry { name, shape, data });
    }
    for suffix in &["proj1.weight", "proj2.weight"] {
        let name = format!("{}{}", enc_prefix, suffix);
        quantize_and_push(ms, &name, &mut quant_tensors);
    }

    // ---- V2: Token embeddings (INT8) ----
    {
        let name = "thinker.model.embed_tokens.weight";
        eprintln!("  Quantizing token embeddings ...");
        quantize_and_push(ms, name, &mut quant_tensors);
    }

    // ---- Write V2 output ----
    let n_quant = quant_tensors.len();
    let n_f32 = f32_tensors.len();
    let n_bf16 = bf16_tensors.len();
    eprintln!(
        "Writing {} INT8 + {} F32 + {} BF16 tensors to {} ...",
        n_quant, n_f32, n_bf16, output_path
    );

    write_qint8_v2_file(output_path, &quant_tensors, &f32_tensors, &bf16_tensors)
        .unwrap_or_else(|e| {
            eprintln!("Failed to write output: {}", e);
            std::process::exit(1);
        });

    let output_size = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "Done! Output: {} ({:.1} MB)",
        output_path,
        output_size as f64 / 1024.0 / 1024.0
    );
}

/// Helper: read a BF16 tensor, quantize to INT8, keep same name.
fn quantize_and_push(ms: &MultiSafetensors, name: &str, quant_tensors: &mut Vec<QuantWriteEntry>) {
    quantize_and_push_rename(ms, name, name, quant_tensors);
}

/// Helper: read a BF16 tensor from `src_name`, quantize to INT8, store as `dst_name`.
fn quantize_and_push_rename(
    ms: &MultiSafetensors,
    src_name: &str,
    dst_name: &str,
    quant_tensors: &mut Vec<QuantWriteEntry>,
) {
    let (_, tmeta) = ms.find(src_name).unwrap_or_else(|| {
        eprintln!("BF16 weight not found: {}", src_name);
        std::process::exit(1);
    });
    let out_dim = tmeta.shape[0] as usize;
    let in_dim = tmeta.shape[1] as usize;

    let bf16_ptr = ms.get_bf16_direct(src_name).unwrap_or_else(|| {
        eprintln!("Failed to get BF16 pointer for {}", src_name);
        std::process::exit(1);
    });

    let (int8_data, scales) = quantize_bf16_to_int8(bf16_ptr, out_dim, in_dim);

    quant_tensors.push(QuantWriteEntry {
        name: dst_name.to_string(),
        shape: vec![out_dim, in_dim],
        int8_data,
        scales,
    });
}
