use crate::ffi::*;
use libc::{self, LC_NUMERIC, free, setlocale, strdup};
use std::{
    ffi::CString,
    io::{BufRead, Write},
    os::raw::{c_char, c_void},
    process::exit,
    ptr::{null, null_mut},
};

// fill the batch with tokens at consecutive positions starting from pos_0, output logits only for the last one
fn batch_set_tokens(
    batch: *mut llama_batch_ext,
    tokens: *const llama_token,
    n_tokens: i32,
    pos_0: llama_pos,
) {
    unsafe { llama_batch_ext_clear(batch) };
    for i in 0..n_tokens {
        let idx: i32 = unsafe { llama_batch_ext_add_token(batch, 0, *tokens.add(i as usize)) };
        let pos: llama_pos = pos_0 + i;
        unsafe { llama_batch_ext_set_pos(batch, idx, &pos) };
    }
    unsafe { llama_batch_ext_set_output_logits(batch, n_tokens - 1, true) };
}

pub struct LlamaCpp {
    model: *mut llama_model,
    messages: Vec<llama_chat_message>,
    formatted: Vec<c_char>,
    prev_len: i32,
    ctx: *mut llama_context,
    vocab: *const llama_vocab,
    batch: *mut llama_batch_ext,
    smpl: *mut llama_sampler,
}

// TODO: A LLM add he following implementation for LlamaCpp.
// It fix an error on the API.
// Check it does and optimize it if needed.
// The native handles are accessed through mutable methods and shared only behind a Mutex.
unsafe impl Send for LlamaCpp {}

impl LlamaCpp {
    pub fn new() -> Self {
        Self {
            model: std::ptr::null_mut(),
            messages: Vec::new(),
            formatted: Vec::new(),
            prev_len: 0,
            ctx: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            batch: std::ptr::null_mut(),
            smpl: std::ptr::null_mut(),
        }
    }

    fn generate(&mut self, prompt: String) -> String {
        let mut response: String = Default::default();
        let is_first: bool =
            unsafe { llama_memory_seq_pos_max(llama_get_memory(self.ctx), 0) == -1 };

        // tokenize the prompt
        let n_prompt_tokens: i32 = unsafe {
            -llama_tokenize(
                self.vocab,
                prompt.as_ptr() as *const i8,
                prompt.len().try_into().unwrap(),
                null_mut(),
                0,
                is_first,
                true,
            )
        };

        let mut prompt_tokens = vec![0; n_prompt_tokens as usize];

        unsafe {
            if llama_tokenize(
                self.vocab,
                prompt.as_ptr() as *const c_char,
                prompt.len() as i32,
                prompt_tokens.as_mut_ptr(),
                prompt_tokens.len() as i32,
                is_first,
                true,
            ) < 0
            {
                // GGML_ABORT("failed to tokenize the prompt\n");
                ggml_abort(
                    c"llm.rs".as_ptr(),
                    line!() as i32,
                    c"failed to tokenize the prompt\n".as_ptr(),
                );
            }
        };

        // the tokens to evaluate next: the prompt, then the sampled token
        let mut tokens: *const llama_token = prompt_tokens.as_mut_ptr();
        let mut n_tokens: i32 = prompt_tokens.len() as i32;
        let mut new_token_id: llama_token;

        loop {
            // check if we have enough space in the context to evaluate this batch
            let n_ctx: u32 = unsafe { llama_n_ctx(self.ctx) };
            let n_ctx_used: llama_pos =
                unsafe { llama_memory_seq_pos_max(llama_get_memory(self.ctx), 0) + 1 };
            if (n_ctx_used + n_tokens) > n_ctx as i32 {
                println!("\033[0m");
                eprintln!("context size exceeded");
                exit(0);
            }

            // positions continue from the memory
            batch_set_tokens(self.batch, tokens, n_tokens, n_ctx_used);

            let ret: i32 =
                unsafe { llama_process(self.ctx, LLAMA_PROCESS_TYPE_DECODE, self.batch) };

            if ret != 0 {
                // GGML_ABORT("failed to decode, ret = %d\n", ret);
                unsafe {
                    ggml_abort(
                        c"llm.rs".as_ptr(),
                        line!() as i32,
                        c"failed to decode, ret = %d\n".as_ptr(),
                        ret,
                    )
                };
            }

            // sample the next token
            new_token_id = unsafe { llama_sampler_sample(self.smpl, self.ctx, -1) };

            // is it an end of generation?
            unsafe {
                if llama_vocab_is_eog(self.vocab, new_token_id) {
                    break;
                }
            };

            // convert the token to a string, print it and add it to the response
            let mut buf: [i8; 256] = [0; 256];
            let n: i32 = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    new_token_id,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n < 0 {
                // GGML_ABORT("failed to convert token to piece\n");
                unsafe {
                    ggml_abort(
                        c"llm.rs".as_ptr(),
                        line!() as i32,
                        c"failed to convert token to piece\n".as_ptr(),
                        ret,
                    )
                };
            }
            let piece = String::from_utf8_lossy(unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize)
            })
            .into_owned();

            print!("{}", piece);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            response += &piece;

            // prepare the next batch with the sampled token
            tokens = &new_token_id;
            n_tokens = 1;
        }

        return response;
    }

    pub fn load(&mut self, model_path: String) -> i32 {
        unsafe { setlocale(LC_NUMERIC, b"C\0".as_ptr() as *const i8) };
        let ngl: i32 = 99;
        let n_ctx: u32 = 2048;

        unsafe { ggml_backend_load_all() };

        // only print errors
        // unsafe {
        //     llama_log_set(
        //         ,
        //         std::ptr::null()
        //     );
        // };

        unsafe { llama_backend_init() };

        // initialize the model
        let mut model_params: llama_model_params = unsafe { llama_model_default_params() };
        model_params.n_gpu_layers = ngl;

        self.model =
            unsafe { llama_model_load_from_file(model_path.as_ptr() as *const i8, model_params) };
        if self.model.is_null() {
            eprintln!("error: unable to load model");
        }

        self.vocab = unsafe { llama_model_get_vocab(self.model) };

        // initialize the context
        let mut ctx_params: llama_context_params = unsafe { llama_context_default_params() };
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx;

        self.ctx = unsafe { llama_init_from_model(self.model, ctx_params) };
        if self.ctx.is_null() {
            eprintln!("error: failed to create the llama_context");
        }

        // initialize the sampler
        self.smpl = unsafe { llama_sampler_chain_init(llama_sampler_chain_default_params()) };
        unsafe { llama_sampler_chain_add(self.smpl, llama_sampler_init_min_p(0.05_f32, 1)) };
        unsafe { llama_sampler_chain_add(self.smpl, llama_sampler_init_temp(0.8f32)) };
        unsafe { llama_sampler_chain_add(self.smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED)) };

        self.batch = unsafe { llama_batch_ext_init(self.ctx) };

        self.messages = Vec::new();
        self.formatted = vec![0; unsafe { llama_n_ctx(self.ctx) } as usize];
        self.prev_len = 0;
        return 0;
    }

    pub fn complete(&mut self, prompt: String) -> String {
        let tmpl: *const c_char = unsafe { llama_model_chat_template(self.model, null()) };

        // add the user input to the message list and format it
        self.messages.push(llama_chat_message {
            role: c"user".as_ptr(),
            content: unsafe { strdup(CString::new(prompt).unwrap().as_ptr()) },
        });
        let mut new_len: i32 = unsafe {
            llama_chat_apply_template(
                tmpl,
                self.messages.as_ptr(),
                self.messages.len(),
                true,
                self.formatted.as_mut_ptr(),
                self.formatted.len() as i32,
            )
        };
        if new_len > self.formatted.len() as i32 {
            self.formatted.resize(new_len as usize, 0);
            new_len = unsafe {
                llama_chat_apply_template(
                    tmpl,
                    self.messages.as_ptr(),
                    self.messages.len(),
                    true,
                    self.formatted.as_mut_ptr(),
                    self.formatted.len() as i32,
                )
            };
        }
        if new_len < 0 {
            eprintln!("failed to apply the chat template");
            return 1.to_string();
        }

        // remove previous messages to obtain the prompt to generate the response
        // std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
        let prompt: String = {
            let bytes: Vec<u8> = self.formatted[self.prev_len as usize..new_len as usize]
                .iter()
                .map(|&c| c as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };

        // generate a response
        print!("\033[33m");
        let response: String = self.generate(prompt.clone());
        print!("\n\033[0m");

        // add the response to the messages
        self.messages.push(llama_chat_message {
            role: c"assistant".as_ptr(),
            content: unsafe { strdup(CString::new(response).unwrap().as_ptr()) },
        });
        self.prev_len = unsafe {
            llama_chat_apply_template(
                tmpl,
                self.messages.as_ptr(),
                self.messages.len(),
                false,
                null_mut(),
                0,
            )
        };
        if self.prev_len < 0 {
            eprintln!("failed to apply the chat template");
            return 1.to_string();
        }
        return 0.to_string();
    }

    pub fn interactive(&mut self) {
        loop {
            // get user input
            print!("\x1b[32m> \x1b[0m");
            std::io::stdout().flush().unwrap();

            let mut prompt = String::new();
            std::io::stdin()
                .lock()
                .read_line(&mut prompt)
                .expect("Failed to read line");

            if prompt.is_empty() {
                break;
            }

            println!("{}", self.complete(prompt));
        }
    }
}

impl Drop for LlamaCpp {
    fn drop(&mut self) {
        // free resources
        for msg in self.messages.iter() {
            unsafe { free(msg.content as *mut c_void) };
        }

        unsafe {
            if !self.batch.is_null() {
                llama_batch_ext_free(self.batch);
            }

            if !self.smpl.is_null() {
                llama_sampler_free(self.smpl);
            }

            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }

            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        };
    }
}
