use std::ffi::{CStr, CString};
use std::os::raw::c_char;

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

unsafe extern "C" fn log_callback(
    level: ggml_log_level,
    text: *const c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
    if level >= ggml_log_level_GGML_LOG_LEVEL_ERROR && !text.is_null() {
        let s = CStr::from_ptr(text).to_string_lossy();
        eprint!("{s}");
    }
}

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

/// Normalize a vector to unit length (L2 normalization).
fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vec.to_vec();
    }
    vec.iter().map(|x| x / norm).collect()
}

/// Generate embeddings for the given input texts using a GGUF embedding model.
pub fn run_embedding(
    model_path: &str,
    inputs: &[String],
    n_gpu_layers: i32,
) -> Result<Vec<Vec<f32>>> {
    unsafe { llama_log_set(Some(log_callback), std::ptr::null_mut()) };
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
    let n_embd = unsafe { llama_model_n_embd(model.0) } as usize;

    let mut ctx_params = unsafe { llama_context_default_params() };
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.embeddings = true;

    let raw_ctx = unsafe { llama_init_from_model(model.0, ctx_params) };
    if raw_ctx.is_null() {
        bail!("failed to create llama_context");
    }
    let ctx = Context(raw_ctx);

    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

    for input in inputs {
        let tokens = tokenize(vocab, input);

        if tokens.is_empty() {
            all_embeddings.push(vec![0.0; n_embd]);
            continue;
        }

        // Clear KV cache for each new input
        unsafe { llama_memory_clear(llama_get_memory(ctx.0), true) };

        let batch = unsafe {
            llama_batch_get_one(tokens.as_ptr() as *mut i32, tokens.len() as i32)
        };

        let ret = unsafe { llama_decode(ctx.0, batch) };
        if ret != 0 {
            bail!("failed to decode input for embedding, ret = {}", ret);
        }

        // Get pooled embeddings (sequence-level) or fall back to last-token embeddings
        let embd_ptr = unsafe { llama_get_embeddings_seq(ctx.0, 0) };
        let embd_ptr = if embd_ptr.is_null() {
            // Fall back to getting embeddings for the last token
            unsafe { llama_get_embeddings_ith(ctx.0, -1) }
        } else {
            embd_ptr
        };

        if embd_ptr.is_null() {
            bail!("failed to get embeddings from model");
        }

        let raw_embd = unsafe { std::slice::from_raw_parts(embd_ptr, n_embd) };
        all_embeddings.push(normalize(raw_embd));
    }

    Ok(all_embeddings)
}
