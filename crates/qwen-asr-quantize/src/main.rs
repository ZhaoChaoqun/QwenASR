//! Offline quantization tool: BF16 safetensors → INT8 `.qint8`
//!
//! Usage:
//!     qwen-asr-quantize <model_dir> [output_path]
//!
//! Reads `model*.safetensors` from `model_dir`, quantizes decoder weights to
//! per-channel symmetric INT8, packs encoder weights and embeddings, and writes
//! a self-contained V2 `.qint8` file.

use qwen_asr::config::QwenConfig;
use qwen_asr::quantize::{
    quantize_bf16_to_int8, write_qint8_v2_file,
    BF16WriteEntry, F32WriteEntry, QuantWriteEntry,
};
use qwen_asr::safetensors::MultiSafetensors;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: qwen-asr-quantize <model_dir> [output_path]");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let output_path = if args.len() >= 3 {
        args[2].clone()
    } else {
        format!("{}/model_int8.qint8", model_dir)
    };

    eprintln!("Loading safetensors from {} ...", model_dir);
    let ms = MultiSafetensors::open(model_dir).unwrap_or_else(|| {
        eprintln!("Failed to open safetensors in {}", model_dir);
        std::process::exit(1);
    });

    // Detect model variant
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

        // Linear weight tensors to quantize (2D matrices)
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
            let (_, tmeta) = ms.find(wname).unwrap_or_else(|| {
                eprintln!("\nWeight not found: {}", wname);
                std::process::exit(1);
            });
            let out_dim = tmeta.shape[0] as usize;
            let in_dim = tmeta.shape[1] as usize;

            let bf16_ptr = ms.get_bf16_direct(wname).unwrap_or_else(|| {
                eprintln!("\nFailed to get BF16 pointer for {}", wname);
                std::process::exit(1);
            });

            let (int8_data, scales) = quantize_bf16_to_int8(bf16_ptr, out_dim, in_dim);

            quant_tensors.push(QuantWriteEntry {
                name: wname.clone(),
                shape: vec![out_dim, in_dim],
                int8_data,
                scales,
            });
        }

        // Norm tensors: keep as f32 (1D, small)
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
    let mut bf16_tensors: Vec<BF16WriteEntry> = Vec::new();

    let enc_prefix = "thinker.audio_tower.";
    eprintln!("  Packing encoder weights (V2 self-contained) ...");

    // Conv layers (F32 - stored as raw f32)
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

    // conv_out.weight (BF16)
    {
        let name = format!("{}conv_out.weight", enc_prefix);
        pack_bf16(&ms, &name, &mut bf16_tensors);
    }

    // Encoder transformer layers
    for i in 0..cfg.enc_layers {
        let lp = format!("{}layers.{}", enc_prefix, i);
        eprint!("  Packing encoder layer {}/{} ...\r", i + 1, cfg.enc_layers);

        // BF16 weight matrices
        for suffix in &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.out_proj.weight",
            "fc1.weight",
            "fc2.weight",
        ] {
            let name = format!("{}.{}", lp, suffix);
            pack_bf16(&ms, &name, &mut bf16_tensors);
        }

        // F32 biases and norms
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
    eprintln!("  Packed {} encoder layers.                ", cfg.enc_layers);

    // ln_post, proj1, proj2
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
        pack_bf16(&ms, &name, &mut bf16_tensors);
    }

    // ---- V2: Token embeddings (BF16) ----
    {
        let name = "thinker.model.embed_tokens.weight";
        eprintln!("  Packing token embeddings ...");
        pack_bf16(&ms, name, &mut bf16_tensors);
    }

    // ---- Write V2 output ----
    let n_quant = quant_tensors.len();
    let n_f32 = f32_tensors.len();
    let n_bf16 = bf16_tensors.len();
    eprintln!(
        "Writing {} INT8 + {} F32 + {} BF16 tensors to {} ...",
        n_quant, n_f32, n_bf16, output_path
    );

    write_qint8_v2_file(&output_path, &quant_tensors, &f32_tensors, &bf16_tensors)
        .unwrap_or_else(|e| {
            eprintln!("Failed to write output: {}", e);
            std::process::exit(1);
        });

    // Print size comparison
    let output_size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "Done! Output: {} ({:.1} MB)",
        output_path,
        output_size as f64 / 1024.0 / 1024.0
    );
}

/// Helper: read a BF16 tensor from safetensors and add as BF16WriteEntry.
fn pack_bf16(ms: &MultiSafetensors, name: &str, bf16_tensors: &mut Vec<BF16WriteEntry>) {
    let (si, tmeta) = ms.find(name).unwrap_or_else(|| {
        eprintln!("BF16 weight not found: {}", name);
        std::process::exit(1);
    });
    let bf16_ptr = ms.shards[si].get_bf16_direct(tmeta).unwrap_or_else(|| {
        eprintln!("Failed to get BF16 pointer for {}", name);
        std::process::exit(1);
    });
    let n = tmeta.numel();
    let data = unsafe { std::slice::from_raw_parts(bf16_ptr, n) }.to_vec();
    let shape: Vec<usize> = tmeta.shape.iter().map(|&d| d as usize).collect();
    bf16_tensors.push(BF16WriteEntry {
        name: name.to_string(),
        shape,
        data,
    });
}
