use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::io::{self, BufRead, Write};

use anyhow::{bail, Result};

use crate::ffi::*;
use crate::mcp::client::{MCPClient, McpConfig};
use crate::tools;
use crate::tui::input::Input;


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

    let is_first =
        unsafe { llama_memory_seq_pos_max(llama_get_memory(ctx.0), 0) == -1 };

    let prompt_tokens = tokenize(vocab, prompt, is_first);

    let mut batch = unsafe {
        llama_batch_get_one(prompt_tokens.as_ptr() as *mut i32, prompt_tokens.len() as i32)
    };

    loop {
        let n_ctx = unsafe { llama_n_ctx(ctx.0) } as i32;
        let n_used = unsafe { llama_memory_seq_pos_max(llama_get_memory(ctx.0), 0) } + 1;
        if n_used + batch.n_tokens > n_ctx {
            eprintln!("\ncontext size exceeded");
            std::process::exit(0);
        }

        let ret = unsafe { llama_decode(ctx.0, batch) };
        if ret != 0 {
            panic!("failed to decode, ret = {ret}");
        }

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

        batch = unsafe { llama_batch_get_one(&new_token as *const i32 as *mut i32, 1) };
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

unsafe extern "C" fn log_callback(
    level: ggml_log_level,
    text: *const ::std::os::raw::c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
    if level >= ggml_log_level_GGML_LOG_LEVEL_ERROR && !text.is_null() {
        let s = CStr::from_ptr(text).to_string_lossy();
        eprint!("{s}");
    }
}

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
    let tools_str =
        serde_json::to_string_pretty(&tools_json).unwrap_or_else(|_| "[]".into());
    format!("\n\n# MCP server tools\n\n{tools_str}")
}

pub async fn run_chat(model_path: &str, n_ctx: u32, n_gpu_layers: i32) -> Result<()> {
    unsafe { llama_log_set(Some(log_callback), std::ptr::null_mut()) };

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

    let mut model_params = unsafe { llama_model_default_params() };
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.progress_callback = None; // Some(progress_cb);

    let c_path = CString::new(model_path).expect("model path contains null byte");
    let raw_model = unsafe { llama_model_load_from_file(c_path.as_ptr(), model_params) };
    if raw_model.is_null() {
        bail!("unable to load model: {}", model_path);
    }
    let model = Model(raw_model);

    let vocab = unsafe { llama_model_get_vocab(model.0) };

    let mut ctx_params = unsafe { llama_context_default_params() };
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

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
                eprintln!("warning: could not connect to MCP server '{}': {e}", entry.name);
            }
        }
    }

    let mcp_prompt_section = build_mcp_tools_prompt(&mcp_tools);
    let system_prompt = format!("{}{}", tools::build_system_prompt(&registry), mcp_prompt_section);

    let mut history: Vec<(String, String)> = Vec::new();

    history.push(("system".into(), system_prompt));

    let tmpl = unsafe { llama_model_chat_template(model.0, std::ptr::null()) };
    let stdin = io::stdin();
    let mut prev_len: i32 = 0;

    'outer: loop {
        let user_input = match Input::read_input() {
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
        }

        history.push(("user".into(), user_input));

        loop {
            let mut c_messages: Vec<llama_chat_message> = Vec::new();
            let mut owned: Vec<(CString, CString)> = Vec::new();

            for (role, content) in &history {
                let role_cs = CString::new(role.as_str()).expect("role contains null byte");
                let content_cs =
                    CString::new(content.as_str()).expect("content contains null byte");
                c_messages.push(llama_chat_message {
                    role: role_cs.as_ptr(),
                    content: content_cs.as_ptr(),
                });
                owned.push((role_cs, content_cs));
            }

            let formatted = apply_template(tmpl, &c_messages, true).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });

            let new_len = formatted.iter().position(|&b| b == 0).unwrap_or(formatted.len()) as i32;
            let prompt = String::from_utf8_lossy(&formatted[prev_len as usize..new_len as usize])
                .to_owned();

            io::stdout().flush().ok();
            let response = generate(&ctx, vocab, &smpl, &prompt, false);

            history.push(("assistant".into(), response.clone()));

            {
                let mut c_msgs2: Vec<llama_chat_message> = Vec::new();
                let mut owned2: Vec<(CString, CString)> = Vec::new();
                for (role, content) in &history {
                    let role_cs = CString::new(role.as_str()).expect("role contains null byte");
                    let content_cs =
                        CString::new(content.as_str()).expect("content contains null byte");
                    c_msgs2.push(llama_chat_message {
                        role: role_cs.as_ptr(),
                        content: content_cs.as_ptr(),
                    });
                    owned2.push((role_cs, content_cs));
                }
                prev_len = unsafe {
                    llama_chat_apply_template(
                        tmpl,
                        c_msgs2.as_ptr(),
                        c_msgs2.len(),
                        false,
                        std::ptr::null_mut(),
                        0,
                    )
                };
                if prev_len < 0 {
                    bail!("failed to apply the chat template");
                }
            }

            let tool_calls = tools::parse_tool_calls(&response);
            if tool_calls.is_empty() {
                break;
            }

            for call in &tool_calls {
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

                // Try standard tools first, then MCP tools
                let result = if let Some(tool) = registry.find(&call.name) {
                    match tool.execute(call.arguments.clone()) {
                        Ok(output) => output,
                        Err(e) => format!("error: {e}"),
                    }
                } else if let Some(mcp_info) = mcp_tools.iter().find(|t| t.tool_name == call.name) {
                    // Find the client that owns this tool
                    if let Some((_name, client)) = mcp_clients.iter().find(|(n, _)| *n == mcp_info.server_name) {
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

        continue 'outer;
    }

    // Clean up MCP clients
    for (_name, client) in mcp_clients {
        let _ = client.cleanup().await;
    }

    Ok(())
}
