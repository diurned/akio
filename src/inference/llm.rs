use std::ffi::CString;
use std::io::{self, BufRead, Write};
use std::os::raw::c_char;

use anyhow::{bail, Result};

use crate::ffi::*;
use crate::mcp::client::{MCPClient, McpConfig};
use crate::tools;
use crate::tui::input::read_input;

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

struct Sampler(*mut llama_sampler);

impl Drop for Sampler {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { llama_sampler_free(self.0) };
        }
    }
}

// struct Backend(*mut llama_backend_free);
//
// impl Drop for Backend {
//   fn drop(&mut self) {
//       if !self.0.is_null() {
//           unsafe { llama_backend_free(self.0) };
//       }
//   }
// }

fn tokenize(vocab: *const llama_vocab, prompt: &str, is_first: bool) -> Vec<llama_token> {
    let c_prompt = CString::new(prompt).expect("prompt contains null byte");
    let n = unsafe {
        -llama_tokenize(
            vocab,
            c_prompt.as_ptr(),
            prompt.len() as i32,
            std::ptr::null_mut(),
            0,
            is_first,
            true,
        )
    };
    let mut tokens = vec![0i32; n as usize];
    let ret = unsafe {
        llama_tokenize(
            vocab,
            c_prompt.as_ptr(),
            prompt.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            is_first,
            true,
        )
    };
    if ret < 0 {
        panic!("failed to tokenize the prompt");
    }
    tokens
}

fn token_to_piece(vocab: *const llama_vocab, token: llama_token) -> String {
    let mut buf = vec![0u8; 256];
    let n = unsafe {
        llama_token_to_piece(
            vocab,
            token,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
            0,
            true,
        )
    };
    if n < 0 {
        panic!("failed to convert token to piece");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

fn generate(
    ctx: &Context,
    vocab: *const llama_vocab,
    smpl: &Sampler,
    prompt: &str,
    silent: bool,
) -> String {
    let mut response = String::new();

    let is_first = unsafe { llama_memory_seq_pos_max(llama_get_memory(ctx.0), 0) == -1 };

    let prompt_tokens = tokenize(vocab, prompt, is_first);

    let n_ctx = unsafe { llama_n_ctx(ctx.0) } as i32;
    let n_batch = unsafe { llama_n_batch(ctx.0) } as usize;

    let check_ctx = |n_new: i32| {
        let n_used = unsafe { llama_memory_seq_pos_max(llama_get_memory(ctx.0), 0) } + 1;
        if n_used + n_new > n_ctx {
            eprintln!("\ncontext size exceeded");
            std::process::exit(0);
        }
    };

    // llama_batch_get_one only sets logits=true on the last token of
    // whatever slice it's given, so decoding chunk-by-chunk and sampling
    // only after the final chunk is correct.
    let mut offset = 0usize;
    while offset < prompt_tokens.len() {
        let end = (offset + n_batch).min(prompt_tokens.len());
        let chunk = &prompt_tokens[offset..end];

        check_ctx(chunk.len() as i32);

        let batch = unsafe { llama_batch_get_one(chunk.as_ptr() as *mut i32, chunk.len() as i32) };
        let ret = unsafe { llama_decode(ctx.0, batch) };
        if ret != 0 {
            panic!("failed to decode, ret = {ret}");
        }
        offset = end;
    }

    loop {
        let new_token = unsafe { llama_sampler_sample(smpl.0, ctx.0, -1) };
        if unsafe { llama_vocab_is_eog(vocab, new_token) } {
            break;
        }
        let piece = token_to_piece(vocab, new_token);
        if !silent {
            print!("{piece}");
            io::stdout().flush().ok();
        }
        response.push_str(&piece);

        check_ctx(1);
        let batch = unsafe { llama_batch_get_one(&new_token as *const i32 as *mut i32, 1) };
        let ret = unsafe { llama_decode(ctx.0, batch) };
        if ret != 0 {
            panic!("failed to decode, ret = {ret}");
        }
    }

    response
}

fn apply_template(
    tmpl: *const c_char,
    messages: &[llama_chat_message],
    add_ass: bool,
) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; 4096];
    let new_len = unsafe {
        llama_chat_apply_template(
            tmpl,
            messages.as_ptr(),
            messages.len(),
            add_ass,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
        )
    };
    if new_len > buf.len() as i32 {
        buf.resize(new_len as usize, 0);
        let new_len2 = unsafe {
            llama_chat_apply_template(
                tmpl,
                messages.as_ptr(),
                messages.len(),
                add_ass,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        if new_len2 < 0 {
            return Err("failed to apply the chat template".into());
        }
    } else if new_len < 0 {
        return Err("failed to apply the chat template".into());
    }
    Ok(buf)
}

fn build_prompt(
    tmpl: *const c_char,
    history: &[(String, String)],
    prev_len: i32,
) -> Result<(String, i32)> {
    let mut c_messages: Vec<llama_chat_message> = Vec::new();
    let mut owned: Vec<(CString, CString)> = Vec::new();
    for (role, content) in history {
        let role_cs = CString::new(role.as_str()).expect("role contains null byte");
        let content_cs = CString::new(content.as_str()).expect("content contains null byte");
        c_messages.push(llama_chat_message {
            role: role_cs.as_ptr(),
            content: content_cs.as_ptr(),
        });
        owned.push((role_cs, content_cs));
    }

    let formatted = apply_template(tmpl, &c_messages, true).map_err(|e| anyhow::anyhow!(e))?;

    let new_len = formatted
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(formatted.len()) as i32;
    let prompt_str =
        String::from_utf8_lossy(&formatted[prev_len as usize..new_len as usize]).into_owned();
    Ok((prompt_str, new_len))
}

fn update_prev_len(tmpl: *const c_char, history: &[(String, String)]) -> Result<i32> {
    let mut c_messages: Vec<llama_chat_message> = Vec::new();
    let mut owned: Vec<(CString, CString)> = Vec::new();
    for (role, content) in history {
        let role_cs = CString::new(role.as_str()).expect("role contains null byte");
        let content_cs = CString::new(content.as_str()).expect("content contains null byte");
        c_messages.push(llama_chat_message {
            role: role_cs.as_ptr(),
            content: content_cs.as_ptr(),
        });
        owned.push((role_cs, content_cs));
    }
    let prev_len = unsafe {
        llama_chat_apply_template(
            tmpl,
            c_messages.as_ptr(),
            c_messages.len(),
            false,
            std::ptr::null_mut(),
            0,
        )
    };
    if prev_len < 0 {
        bail!("failed to apply the chat template");
    }
    Ok(prev_len)
}

async fn execute_tool_calls(
    tool_calls: &[tools::ToolCall],
    registry: &tools::ToolRegistry,
    mcp_tools: &[McpToolInfo],
    mcp_clients: &[(String, MCPClient)],
    history: &mut Vec<(String, String)>,
) {
    let stdin = io::stdin();
    for call in tool_calls {
        let args_display = serde_json::to_string_pretty(&call.arguments)
            .unwrap_or_else(|_| call.arguments.to_string());

        println!(
            "\n\x1b[91m\n[tool call] {} ({})\x1b[0m",
            call.name, args_display
        );
        print!("\x1b[91mRun this tool? [y/N] \x1b[0m");
        io::stdout().flush().ok();

        let mut approval = String::new();
        match stdin.lock().read_line(&mut approval) {
            Ok(0) | Err(_) => {
                println!("\x1b[31m[tool skipped — no input]\x1b[0m");
                history.push((
                    "tool".into(),
                    format!("error: user declined to run tool `{}`", call.name),
                ));
                continue;
            }
            Ok(_) => {}
        }

        let approved = approval.trim().eq_ignore_ascii_case("y");
        if !approved {
            println!("\x1b[31m[tool skipped by user]\x1b[0m");
            history.push((
                "tool".into(),
                format!("error: user declined to run tool `{}`", call.name),
            ));
            continue;
        }

        let result = if let Some(tool) = registry.find(&call.name) {
            match tool.execute(call.arguments.clone()) {
                Ok(output) => output,
                Err(e) => format!("error: {e}"),
            }
        } else if let Some(mcp_info) = mcp_tools.iter().find(|t| t.tool_name == call.name) {
            if let Some((_name, client)) =
                mcp_clients.iter().find(|(n, _)| *n == mcp_info.server_name)
            {
                match client.call_tool(&call.name, call.arguments.clone()).await {
                    Ok(output) => output,
                    Err(e) => format!("error: {e}"),
                }
            } else {
                format!("error: MCP server '{}' not connected", mcp_info.server_name)
            }
        } else {
            format!("error: unknown tool `{}`", call.name)
        };

        println!("\n\x1b[35m[tool result]\n{result}\x1b[0m");

        history.push(("tool".into(), result));
    }
}

use crate::inference::log as llama_log;

/// Describes an MCP tool for inclusion in the system prompt.
struct McpToolInfo {
    server_name: String,
    tool_name: String,
    description: String,
    input_schema: serde_json::Value,
}

/// Build system prompt section for MCP tools.
fn build_mcp_tools_prompt(mcp_tools: &[McpToolInfo]) -> String {
    if mcp_tools.is_empty() {
        return String::new();
    }
    let mut tools_json: Vec<serde_json::Value> = Vec::new();
    for t in mcp_tools {
        tools_json.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": t.tool_name,
                "description": t.description,
                "parameters": t.input_schema
            }
        }));
    }
    let tools_str = serde_json::to_string_pretty(&tools_json).unwrap_or_else(|_| "[]".into());
    format!("\n\n# MCP server tools\n\n{tools_str}")
}

/// Clamps the requested GPU layer count down to what should fit in free
/// device memory, using only the model file's on-disk size and a backend
/// memory query — deliberately NOT touching llama.cpp's model loader.
///
/// An earlier version of this function did a "probe load" with
/// `no_alloc = true` to get an exact tensor byte count and layer count
/// straight from llama.cpp. That hits `GGML_ASSERT(!ml.no_alloc)` inside
/// `load_tensors()` on current llama.cpp — `no_alloc` isn't a supported way
/// to stop `llama_model_load_from_file` short of actually loading tensors,
/// despite what the field's doc-comment suggests. Rather than guess at the
/// undocumented "correct" way to invoke it, this version avoids the model
/// loader entirely, so it can't hit that (or a similar) assertion:
///
/// - model size comes from `std::fs::metadata`, not from llama.cpp
/// - free device memory comes from `ggml_backend_dev_memory`, which is pure
///   hardware introspection and has no per-model state to violate
///
/// The tradeoff is precision: file size is a proxy for in-memory footprint
/// (close enough for mmap'd, roughly-contiguous GGUF weights, but not
/// exact), and without a real layer count, a partial fit is estimated by
/// scaling the request rather than counting exact layers. Context-size
/// auto-clamping has been dropped for now rather than built on the same
/// kind of guesswork that just broke here — flash attention + the Q8_0 KV
/// cache below already cut KV memory substantially, but if you still see
/// context-related OOM after this, lower `--n-ctx` manually for now.
fn fit_to_available_memory(model_path: &str, requested_gpu_layers: i32) -> i32 {
    let model_bytes = match std::fs::metadata(model_path) {
        Ok(meta) => meta.len(),
        Err(_) => return requested_gpu_layers, // let the real load report the error
    };

    // Find the first GPU-ish device (covers both discrete GPUs and Apple
    // Silicon's integrated/unified-memory Metal device) and read its free
    // memory.
    let mut free_bytes: Option<u64> = None;
    unsafe {
        for i in 0..ggml_backend_dev_count() {
            let dev = ggml_backend_dev_get(i);
            let kind = ggml_backend_dev_type(dev);
            if kind == ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                || kind == ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                let mut free = 0usize;
                let mut total = 0usize;
                ggml_backend_dev_memory(dev, &mut free, &mut total);
                free_bytes = Some(free as u64);
                break;
            }
        }
    }

    let Some(free) = free_bytes else {
        return requested_gpu_layers; // no GPU-ish device found — nothing to clamp
    };

    const SAFETY_MARGIN: u64 = 1024 * 1024 * 1024; // keep 1 GiB free for scratch/compute buffers
    let budget = free.saturating_sub(SAFETY_MARGIN);

    if budget >= model_bytes {
        return requested_gpu_layers; // whole model should fit as requested
    }

    if requested_gpu_layers < 0 {
        // "offload everything" was requested but the whole model doesn't
        // fit, and there's no layer count to split it by — fail safe to
        // CPU rather than guess a specific layer count.
        eprintln!(
            "warning: model (~{} MiB) likely won't fit in ~{} MiB free device memory; falling back to CPU-only",
            model_bytes / (1024 * 1024),
            free / (1024 * 1024)
        );
        return 0;
    }

    // Scale the requested layer count down by roughly how much of the model
    // fits. This assumes weights are ~evenly sized across layers, which is
    // an approximation (embedding/output layers are often larger), not an
    // exact fit — but it fails toward "less GPU memory used", which is the
    // safe direction.
    let ratio = (budget as f64 / model_bytes as f64).clamp(0.0, 1.0);
    let adjusted = ((requested_gpu_layers as f64) * ratio).floor() as i32;
    if adjusted < requested_gpu_layers {
        eprintln!(
            "warning: {requested_gpu_layers} GPU layers requested but only ~{} MiB free; offloading ~{adjusted} layers instead",
            free / (1024 * 1024)
        );
    }
    adjusted.max(0)
}

pub async fn run_chat(
    model_path: &str,
    n_ctx: u32,
    b_ctx: u32,
    n_gpu_layers: i32,
    verbose: &str,
    prompt: Option<&str>,
) -> Result<()> {
    llama_log::set_min_level(crate::utils::log::parse_log_level(verbose));
    unsafe { llama_log_set(Some(llama_log::log_callback), std::ptr::null_mut()) };

    unsafe { ggml_backend_load_all() };

    // unsafe extern "C" fn progress_cb(progress: f32, _user_data: *mut ::std::os::raw::c_void) -> bool {
    //     let pct = (progress * 100.0) as u32;
    //     static LAST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    //     let last = LAST.load(std::sync::atomic::Ordering::Relaxed);
    //     if pct > last {
    //         LAST.store(pct, std::sync::atomic::Ordering::Relaxed);
    //         print!("\r\x1b[2KLoading model: [{:<50}] {}%",
    //             "#".repeat((pct as usize * 50 / 100).min(50)),
    //             pct);
    //         io::stdout().flush().ok();
    //         if pct >= 100 {
    //             println!();
    //             LAST.store(0, std::sync::atomic::Ordering::Relaxed);
    //         }
    //     }
    //     true
    // }

    let c_path = CString::new(model_path).expect("model path contains null byte");

    // Crash prevention: clamp the requested GPU layers down to what will
    // actually fit in free device memory before attempting the real load.
    // See fit_to_available_memory() for why this only uses file size + a
    // backend memory query rather than probing through llama.cpp itself.
    let n_gpu_layers = fit_to_available_memory(model_path, n_gpu_layers);

    let mut model_params = unsafe { llama_model_default_params() };
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.progress_callback = None; // Some(progress_cb);

    let mut ctx_params = unsafe { llama_context_default_params() };
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = b_ctx;

    // Performance: flash attention + quantized KV cache cut memory traffic
    // per token and roughly halve KV cache size; quantized K/V requires
    // flash attention to be on.
    ctx_params.flash_attn_type = llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.type_k = ggml_type_GGML_TYPE_Q8_0;
    ctx_params.type_v = ggml_type_GGML_TYPE_Q8_0;

    // Performance: use all available CPU threads instead of the library's
    // conservative built-in default.
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    let raw_model = unsafe { llama_model_load_from_file(c_path.as_ptr(), model_params) };
    if raw_model.is_null() {
        bail!("unable to load model: {}", model_path);
    }
    let model = Model(raw_model);

    let vocab = unsafe { llama_model_get_vocab(model.0) };

    let raw_ctx = unsafe { llama_init_from_model(model.0, ctx_params) };
    if raw_ctx.is_null() {
        bail!("failed to create llama_context");
    }
    let ctx = Context(raw_ctx);

    let raw_smpl = unsafe { llama_sampler_chain_init(llama_sampler_chain_default_params()) };
    unsafe {
        llama_sampler_chain_add(raw_smpl, llama_sampler_init_min_p(0.05, 1));
        llama_sampler_chain_add(raw_smpl, llama_sampler_init_temp(0.8));
        llama_sampler_chain_add(raw_smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    let smpl = Sampler(raw_smpl);

    let registry = tools::default_registry();

    // Connect to registered MCP servers
    let mcp_config = McpConfig::load().unwrap_or_default();
    let mut mcp_clients: Vec<(String, MCPClient)> = Vec::new();
    let mut mcp_tools: Vec<McpToolInfo> = Vec::new();

    for entry in &mcp_config.servers {
        let mut client = MCPClient::new();
        match client.connect_to_server(&entry.command, &entry.args).await {
            Ok(tools) => {
                for tool in &tools {
                    mcp_tools.push(McpToolInfo {
                        server_name: entry.name.clone(),
                        tool_name: tool.name.to_string(),
                        description: tool.description.as_deref().unwrap_or("").to_string(),
                        input_schema: serde_json::to_value(&tool.input_schema).unwrap_or_default(),
                    });
                }
                mcp_clients.push((entry.name.clone(), client));
            }
            Err(e) => {
                eprintln!(
                    "warning: could not connect to MCP server '{}': {e}",
                    entry.name
                );
            }
        }
    }

    let mcp_prompt_section = build_mcp_tools_prompt(&mcp_tools);
    let system_prompt = format!(
        "{}{}",
        tools::build_system_prompt(&registry),
        mcp_prompt_section
    );

    let mut history: Vec<(String, String)> = Vec::new();

    history.push(("system".into(), system_prompt));

    let tmpl = unsafe { llama_model_chat_template(model.0, std::ptr::null()) };
    let mut prev_len: i32 = 0;

    // Non-interactive mode: single prompt, generate response, and exit
    if let Some(prompt_text) = prompt {
        history.push(("user".into(), prompt_text.to_string()));

        loop {
            let (prompt_str, _new_len) = build_prompt(tmpl, &history, prev_len)?;
            let response = generate(&ctx, vocab, &smpl, &prompt_str, false);
            history.push(("assistant".into(), response.clone()));
            prev_len = update_prev_len(tmpl, &history)?;

            let tool_calls = tools::parse_tool_calls(&response);
            if tool_calls.is_empty() {
                break;
            }
            execute_tool_calls(
                &tool_calls,
                &registry,
                &mcp_tools,
                &mcp_clients,
                &mut history,
            )
            .await;
        }

        println!();

        for (_name, client) in mcp_clients {
            let _ = client.cleanup().await;
        }
        return Ok(());
    }

    'outer: loop {
        let user_input = match read_input() {
            Some(s) => s,
            None => break,
        };

        if user_input.is_empty() {
            break;
        } else if user_input.starts_with('/') {
            match user_input.as_str() {
                "/quit" | "/exit" => {
                    unsafe {
                        // Free context (uses the model)
                        llama_free(ctx.0);

                        // Free model
                        llama_model_free(model.0);

                        // Free sampler
                        llama_sampler_free(smpl.0);

                        // Optional: backend cleanup (safe to call once at end)
                        // llama_backend_free();
                    }
                    std::process::exit(0);
                }
                _ => {
                    println!("Unknown command.");
                    println!("\n/exit  Exit Akio");
                    continue 'outer;
                }
            }
        } else if user_input.starts_with('!') {
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(&user_input[1..])
                .output()
                .expect("failed to execute");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            print!("{}", stdout);
            if !stderr.is_empty() {
                eprint!("{}", stderr);
            }

            history.push(("user".into(), user_input));

            let mut command_output = stdout.to_string();
            if !stderr.is_empty() {
                if !command_output.is_empty() {
                    command_output.push('\n');
                }
                command_output.push_str(&stderr);
            }
            if !command_output.is_empty() {
                history.push(("user".into(), command_output));
            }
            continue 'outer;
        }

        history.push(("user".into(), user_input));

        loop {
            let (prompt_str, _new_len) = build_prompt(tmpl, &history, prev_len)?;
            io::stdout().flush().ok();
            let response = generate(&ctx, vocab, &smpl, &prompt_str, false);
            history.push(("assistant".into(), response.clone()));
            prev_len = update_prev_len(tmpl, &history)?;

            let tool_calls = tools::parse_tool_calls(&response);
            if tool_calls.is_empty() {
                break;
            }
            execute_tool_calls(
                &tool_calls,
                &registry,
                &mcp_tools,
                &mcp_clients,
                &mut history,
            )
            .await;
        }

        continue 'outer;
    }

    // Clean up MCP clients
    for (_name, client) in mcp_clients {
        let _ = client.cleanup().await;
    }

    Ok(())
}
