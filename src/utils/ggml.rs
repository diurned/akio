use super::gguf::{parse_gguf, GgufMeta};
use super::memory::get_mem_info;


fn fmt_bytes(b: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    const KIB: u64 = 1024;
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else if b >= MIB {
        format!("{:.2} MiB", b as f64 / MIB as f64)
    } else if b >= KIB {
        format!("{:.2} KiB", b as f64 / KIB as f64)
    } else {
        format!("{b} B")
    }
}

fn kv_bytes_per_element(cache_type: &str) -> f64 {
    match cache_type {
        "q8_0" => 1.0,
        "q4_0" => 0.5,
        "f32" => 4.0,
        _ => 2.0, // f16 default
    }
}

/// Estimate (kv_cache_bytes, graph_bytes) for the given context and batch size.
/// Formulas ported from https://github.com/ollama/ollama/blob/main/fs/ggml/ggml.go
pub fn graph_size(meta: &GgufMeta, context: u64, batch: u64) -> (u64, u64) {
    let embedding = meta.embedding_length();
    let heads = meta.head_count();
    let heads_kv = meta.head_count_kv();
    let emb_head_k = meta.embedding_head_k();
    let emb_head_v = meta.embedding_head_v();
    let vocab = meta.vocab_size();
    let blocks = meta.block_count();

    let bpe = kv_bytes_per_element("f16");

    let kv_per_layer = (context.saturating_mul(emb_head_k.saturating_add(emb_head_v))
        .saturating_mul(heads_kv)) as f64
        * bpe;
    let kv_cache = (kv_per_layer * blocks as f64) as u64;

    let graph: u64 = match meta.architecture().as_str() {
        "llama" | "llama4" => {
            let base = std::cmp::max(
                4u64.saturating_mul(batch)
                    .saturating_mul(1u64.saturating_add(4 * embedding).saturating_add(
                        context.saturating_mul(1u64.saturating_add(heads)),
                    )),
                4u64.saturating_mul(batch).saturating_mul(embedding.saturating_add(vocab)),
            );
            // Mixtral 8×22B
            if meta.find_tensor("blk.0.ffn_gate_exps.weight").is_some() {
                let ff = meta.feed_forward_length();
                4u64.saturating_mul(batch).saturating_mul(
                    2 * ff + heads_kv + embedding + context + emb_head_k * heads_kv,
                )
            // Mixtral 8×7B
            } else if let Some(t) = meta.find_tensor("blk.0.ffn_gate.0.weight") {
                let ff1 = t.dim1();
                std::cmp::max(
                    4u64.saturating_mul(batch).saturating_mul(
                        3 + emb_head_k * heads_kv
                            + embedding
                            + context.saturating_mul(1u64.saturating_add(heads))
                            + ff1,
                    ),
                    base,
                )
            } else {
                base
            }
        }

        // Qwen2 / Qwen3 (same arch key in llama.cpp)
        "qwen2" | "qwen3" | "qwen25" => std::cmp::max(
            4u64.saturating_mul(batch).saturating_mul(embedding.saturating_add(vocab)),
            4u64.saturating_mul(batch).saturating_mul(
                1u64.saturating_add(2 * embedding)
                    .saturating_add(context)
                    .saturating_add(context.saturating_mul(heads)),
            ),
        ),

        "gemma" | "gemma2" | "gemma3" | "gemma3n" => {
            let base = std::cmp::max(
                4u64.saturating_mul(batch).saturating_mul(embedding.saturating_add(vocab)),
                4u64.saturating_mul(batch).saturating_mul(
                    2u64.saturating_add(context)
                        .saturating_add(context.saturating_mul(heads))
                        .saturating_add(2 * embedding)
                        .saturating_add(2 * emb_head_k * heads),
                ),
            );
            if meta.architecture() == "gemma3n" { base.saturating_mul(4) } else { base }
        }

        "phi2" => std::cmp::max(
            4u64.saturating_mul(batch).saturating_mul(embedding.saturating_add(vocab)),
            4u64.saturating_mul(batch).saturating_mul(
                1u64.saturating_add(4 * embedding)
                    .saturating_add(context)
                    .saturating_add(context.saturating_mul(heads)),
            ),
        ),

        "stablelm" => 4u64
            .saturating_mul(batch)
            .saturating_mul(context.saturating_mul(1 + heads).saturating_add(3 * embedding + 2)),

        "command-r" => std::cmp::max(
            4u64.saturating_mul(batch).saturating_mul(embedding.saturating_add(vocab)),
            4u64.saturating_mul(batch).saturating_mul(
                2u64.saturating_add(4 * embedding)
                    .saturating_add(context.saturating_mul(1u64.saturating_add(heads))),
            ),
        ),

        "deepseek2" => std::cmp::max(
            4u64.saturating_mul(batch).saturating_mul(3 * embedding + vocab),
            4u64.saturating_mul(batch).saturating_mul(
                3u64.saturating_mul(embedding)
                    .saturating_add(2)
                    .saturating_add(context.saturating_mul(1u64.saturating_add(heads_kv)))
                    .saturating_add(2 * emb_head_k * heads_kv),
            ),
        ),

        "chatglm" => 4u64.saturating_mul(batch).saturating_mul(embedding.saturating_add(vocab)),

        _ => {
            // fallback: rough activation buffer
            4u64.saturating_mul(batch).saturating_mul(
                2u64.saturating_mul(embedding)
                    .saturating_add(vocab)
                    .saturating_add(context.saturating_mul(heads.max(1))),
            )
        }
    };

    (kv_cache, graph)
}

/// Total memory estimate (bytes) for loading and running the model at the given context size.
/// weights ≈ file size + KV cache + arch-specific graph buffers + 256 MiB overhead
pub fn estimate_memory(path: &str, context_size: u32) -> Result<u64, String> {
    let file_size = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat '{}': {e}", path))?
        .len();

    let meta = parse_gguf(path)?;

    let context = context_size as u64;
    let batch: u64 = 512;

    let (kv_cache, graph) = graph_size(&meta, context, batch);
    let overhead: u64 = 256 * 1024 * 1024; // 256 MiB

    let arch = meta.architecture();
    let blocks = meta.block_count();
    let emb = meta.embedding_length();
    let ctx_limit = meta.context_length();

    eprintln!(
        "model: arch={arch}, layers={blocks}, embedding={emb}, \
         max-context={ctx_limit}, vocab={}",
        meta.vocab_size()
    );
    eprintln!(
        "memory estimate: weights={}, kv-cache={}, graph={}, overhead={}",
        fmt_bytes(file_size),
        fmt_bytes(kv_cache),
        fmt_bytes(graph),
        fmt_bytes(overhead),
    );

    Ok(file_size
        .saturating_add(kv_cache)
        .saturating_add(graph)
        .saturating_add(overhead))
}

/// Returns Err (with an eprintln) if the estimated requirement exceeds available RAM.
pub fn check_memory(path: &str, context_size: u32) -> Result<(), String> {
    let estimated = estimate_memory(path, context_size)?;
    let mem = get_mem_info()?;

    eprintln!(
        "memory: estimated={}, available={}",
        fmt_bytes(estimated),
        fmt_bytes(mem.available),
    );

    if estimated > mem.available {
        eprintln!(
            "\nerror: not enough memory to run this model.\n\
             \n\
             Required  (estimated) : {}\n\
             Available             : {}\n\
             \n\
             Try a smaller model or reduce --context-size.",
            fmt_bytes(estimated),
            fmt_bytes(mem.available),
        );
        return Err("insufficient memory".into());
    }

    Ok(())
}
