use std::path::PathBuf;

fn main() {
    // Build llama.cpp using cmake
    let dst = cmake::Config::new("llama.cpp")
        .build_target("llama")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();

    // Tell cargo where to find the built library
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src/ggml-blas", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src/ggml-metal", dst.display());
    println!("cargo:rustc-link-search=native={}/build/src", dst.display());

    // Link llama and ggml
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    #[cfg(target_os = "macos")]
    {
        // ggml may enable optional BLAS/Metal backends on macOS.
        println!("cargo:rustc-link-lib=static=ggml-blas");
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }

    // Link C++ standard library
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");

    // Link OpenMP (required by ggml-cpu)
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=gomp");

    // Link pthreads
    println!("cargo:rustc-link-lib=pthread");

    // Rerun if llama.cpp changes
    println!("cargo:rerun-if-changed=llama.cpp/include/llama.h");
    println!("cargo:rerun-if-changed=build.rs");

    // Generate bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header("llama.cpp/include/llama.h")
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        // llama.h includes ggml.h — both include dirs are required
        .clang_arg("-Illama.cpp/include")
        .clang_arg("-Illama.cpp/ggml/include")
        // Core model/context/vocab types
        .allowlist_type("llama_model")
        .allowlist_type("llama_context")
        .allowlist_type("llama_vocab")
        .allowlist_type("llama_sampler")
        .allowlist_type("llama_batch")
        .allowlist_type("llama_token")
        .allowlist_type("llama_chat_message")
        .allowlist_type("llama_model_params")
        .allowlist_type("llama_context_params")
        .allowlist_type("llama_sampler_chain_params")
        .allowlist_type("ggml_log_level")
        .allowlist_type("ggml_log_level_.*")
        // Functions
        .allowlist_function("llama_model_default_params")
        .allowlist_function("llama_context_default_params")
        .allowlist_function("llama_model_load_from_file")
        .allowlist_function("llama_model_free")
        .allowlist_function("llama_model_get_vocab")
        .allowlist_function("llama_model_chat_template")
        .allowlist_function("llama_init_from_model")
        .allowlist_function("llama_free")
        .allowlist_function("llama_n_ctx")
        .allowlist_function("llama_get_memory")
        .allowlist_function("llama_memory_seq_pos_max")
        .allowlist_function("llama_decode")
        .allowlist_function("llama_batch_get_one")
        .allowlist_function("llama_tokenize")
        .allowlist_function("llama_token_to_piece")
        .allowlist_function("llama_vocab_is_eog")
        .allowlist_function("llama_chat_apply_template")
        .allowlist_function("llama_sampler_chain_init")
        .allowlist_function("llama_sampler_chain_default_params")
        .allowlist_function("llama_sampler_chain_add")
        .allowlist_function("llama_sampler_init_min_p")
        .allowlist_function("llama_sampler_init_temp")
        .allowlist_function("llama_sampler_init_dist")
        .allowlist_function("llama_sampler_sample")
        .allowlist_function("llama_sampler_free")
        .allowlist_function("llama_log_set")
        .allowlist_function("ggml_backend_load_all")
        .allowlist_var("LLAMA_DEFAULT_SEED")
        .allowlist_var("GGML_LOG_LEVEL_ERROR")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
