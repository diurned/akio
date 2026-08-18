use std::ffi::CString;

use anyhow::{bail, Result};

use crate::ffi::*;

struct Model(*mut llama_model);

impl Drop for Model {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { llama_model_free(self.0) };
        }
    }
}

struct Context(*mut llama_context);

impl Drop for Context {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { llama_free(self.0) };
        }
    }
}

struct Batch(llama_batch);

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.0) };
    }
}

use crate::inference::log as llama_log;

fn tokenize(vocab: *const llama_vocab, text: &str) -> Vec<llama_token> {
    let c_text = CString::new(text).expect("text contains null byte");
    let n = unsafe {
        -llama_tokenize(
            vocab,
            c_text.as_ptr(),
            text.len() as i32,
            std::ptr::null_mut(),
            0,
            true,
            true,
        )
    };
    let mut tokens = vec![0i32; n as usize];
    let ret = unsafe {
        llama_tokenize(
            vocab,
            c_text.as_ptr(),
            text.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,
            true,
        )
    };
    if ret < 0 {
        panic!("failed to tokenize the input");
    }
    tokens
}

/// L2-normalize `src` and write the result into `dst`.
fn normalize_into(src: &[f32], dst: &mut [f32]) {
    let norm: f32 = src.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        dst.copy_from_slice(src);
    } else {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s / norm;
        }
    }
}

/// Append one token to a batch allocated by `llama_batch_init`.
///
/// # Safety
/// `batch.n_tokens` must be less than the capacity passed to `llama_batch_init`.
unsafe fn batch_add(
    batch: &mut llama_batch,
    token: llama_token,
    pos: llama_pos,
    seq_id: llama_seq_id,
    logits: bool,
) {
    let n = batch.n_tokens as isize;
    *batch.token.offset(n) = token;
    *batch.pos.offset(n) = pos;
    *batch.n_seq_id.offset(n) = 1;
    *(*batch.seq_id.offset(n)).offset(0) = seq_id;
    *batch.logits.offset(n) = logits as i8;
    batch.n_tokens += 1;
}

/// Decode `batch`, then write L2-normalized embeddings into `output`.
///
/// `output` is a flat slice indexed as `[embd_pos * n_embd .. (embd_pos+1) * n_embd]`,
/// where `embd_pos` is the sequence id (pooled) or token index (NONE pooling).
unsafe fn batch_decode(
    ctx: *mut llama_context,
    batch: &llama_batch,
    output: &mut [f32],
    n_embd: usize,
    pooling_type: llama_pooling_type,
) -> Result<()> {
    llama_memory_clear(llama_get_memory(ctx), true);

    let ret = llama_decode(ctx, *batch);
    if ret != 0 {
        bail!("llama_decode failed with ret={ret}");
    }

    for i in 0..batch.n_tokens {
        if *batch.logits.offset(i as isize) == 0 {
            continue;
        }

        let (embd_ptr, embd_pos) = if pooling_type == llama_pooling_type_LLAMA_POOLING_TYPE_NONE {
            let p = llama_get_embeddings_ith(ctx, i);
            (p, i as usize)
        } else {
            let seq = *(*batch.seq_id.offset(i as isize)).offset(0);
            let p = llama_get_embeddings_seq(ctx, seq);
            (p, seq as usize)
        };

        if embd_ptr.is_null() {
            bail!("failed to get embeddings at token index {i}");
        }

        let raw = std::slice::from_raw_parts(embd_ptr, n_embd);
        let dst = &mut output[embd_pos * n_embd..(embd_pos + 1) * n_embd];
        normalize_into(raw, dst);
    }

    Ok(())
}

/// Generate embeddings for the given input texts using a GGUF embedding model.
pub fn run_embedding(
    model_path: &str,
    inputs: &[String],
    n_gpu_layers: i32,
) -> Result<Vec<Vec<f32>>> {
    unsafe { llama_log_set(Some(llama_log::log_callback), std::ptr::null_mut()) };
    unsafe { ggml_backend_load_all() };

    let mut model_params = unsafe { llama_model_default_params() };
    model_params.n_gpu_layers = n_gpu_layers;

    let c_path = CString::new(model_path).expect("model path contains null byte");
    let raw_model = unsafe { llama_model_load_from_file(c_path.as_ptr(), model_params) };
    if raw_model.is_null() {
        bail!("unable to load model: {}", model_path);
    }
    let model = Model(raw_model);

    let vocab = unsafe { llama_model_get_vocab(model.0) };
    // Use n_embd_out (may differ from n_embd for classification/rerank models).
    let n_embd = unsafe { llama_model_n_embd_out(model.0) } as usize;

    let n_seq_max = unsafe { llama_max_parallel_sequences() } as u32;

    const N_CTX: u32 = 512;
    const N_BATCH: u32 = 512;

    let mut ctx_params = unsafe { llama_context_default_params() };
    ctx_params.n_ctx = N_CTX;
    ctx_params.n_batch = N_BATCH;
    // For non-causal (encoder) models ubatch must equal batch.
    ctx_params.n_ubatch = N_BATCH;
    ctx_params.n_seq_max = n_seq_max;
    ctx_params.embeddings = true;

    let raw_ctx = unsafe { llama_init_from_model(model.0, ctx_params) };
    if raw_ctx.is_null() {
        bail!("failed to create llama_context");
    }
    let ctx = Context(raw_ctx);

    let pooling_type = unsafe { llama_pooling_type(ctx.0) };
    let n_ctx = unsafe { llama_n_ctx(ctx.0) } as usize;

    // Tokenize all inputs upfront.
    let all_tokens: Vec<Vec<llama_token>> = inputs.iter().map(|s| tokenize(vocab, s)).collect();

    let n_prompts = all_tokens.len();
    let n_batch = N_BATCH as usize;

    // Validate that no single input exceeds the context window or batch capacity.
    for (i, toks) in all_tokens.iter().enumerate() {
        if toks.len() > n_ctx {
            bail!(
                "input {} has {} tokens which exceeds the context size ({})",
                i,
                toks.len(),
                n_ctx
            );
        }
        if toks.len() > n_batch {
            bail!(
                "input {} has {} tokens which exceeds the batch size ({}); \
                 increase N_BATCH or shorten the input",
                i,
                toks.len(),
                n_batch
            );
        }
    }

    // Allocate a flat output buffer.  For NONE pooling there is one row per
    // token; for all other pooling types there is one row per input.
    let n_embd_count = if pooling_type == llama_pooling_type_LLAMA_POOLING_TYPE_NONE {
        all_tokens.iter().map(|t| t.len()).sum::<usize>()
    } else {
        n_prompts
    };
    let mut embeddings = vec![0.0f32; n_embd_count * n_embd];

    // Initialize the batch.
    let raw_batch = unsafe { llama_batch_init(n_batch as i32, 0, n_seq_max as i32) };
    let mut batch = Batch(raw_batch);

    // e = rows written to `embeddings` so far.
    // s = number of sequences loaded into the current (unflushed) batch.
    let mut e: usize = 0;
    let mut s: i32 = 0;

    for k in 0..n_prompts {
        let inp = &all_tokens[k];
        let n_toks = inp.len();

        // Flush the current batch if adding this input would overflow it.
        if (batch.0.n_tokens as usize + n_toks > n_batch) || (s >= n_seq_max as i32) {
            unsafe {
                batch_decode(
                    ctx.0,
                    &batch.0,
                    &mut embeddings[e * n_embd..],
                    n_embd,
                    pooling_type,
                )?;
            }
            e += if pooling_type == llama_pooling_type_LLAMA_POOLING_TYPE_NONE {
                batch.0.n_tokens as usize
            } else {
                s as usize
            };
            s = 0;
            batch.0.n_tokens = 0;
        }

        // Add all tokens of this input with sequence id `s`.
        for (pos, &tok) in inp.iter().enumerate() {
            unsafe {
                batch_add(&mut batch.0, tok, pos as llama_pos, s, true);
            }
        }
        s += 1;
    }

    // Decode the final (possibly partial) batch.
    if batch.0.n_tokens > 0 {
        unsafe {
            batch_decode(
                ctx.0,
                &batch.0,
                &mut embeddings[e * n_embd..],
                n_embd,
                pooling_type,
            )?;
        }
    }

    // Build the per-input result.
    let result = if pooling_type == llama_pooling_type_LLAMA_POOLING_TYPE_NONE {
        // Aggregate per-token embeddings into one embedding per input via mean
        // pooling followed by L2 normalization.
        let mut result = Vec::with_capacity(n_prompts);
        let mut offset = 0usize;
        for toks in &all_tokens {
            let n = toks.len();
            if n == 0 {
                result.push(vec![0.0f32; n_embd]);
                continue;
            }
            let mut mean = vec![0.0f32; n_embd];
            for i in 0..n {
                let row = &embeddings[(offset + i) * n_embd..(offset + i + 1) * n_embd];
                for (m, v) in mean.iter_mut().zip(row.iter()) {
                    *m += v;
                }
            }
            let scale = 1.0 / n as f32;
            for m in &mut mean {
                *m *= scale;
            }
            // Re-normalize after averaging.
            let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for m in &mut mean {
                    *m /= norm;
                }
            }
            result.push(mean);
            offset += n;
        }
        Ok(result)
    } else {
        // One normalized embedding per input, already laid out sequentially.
        Ok(embeddings.chunks(n_embd).map(|c| c.to_vec()).collect())
    };

    // Explicit cleanup in reverse-construction order, mirroring the C example's
    // llama_batch_free / llama_free / llama_model_free calls.
    drop(batch);
    drop(ctx);
    drop(model);

    return result;
}
