use std::path::PathBuf;

fn main() {
    // Build llama.cpp using cmake
    let mut cmake_config = cmake::Config::new("llama.cpp");
    cmake_config
        .build_target("llama")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF");

    // On MSVC the cmake crate always passes /MD (release CRT) regardless of the
    // Cargo profile, but a cmake Debug configuration still defines _DEBUG which
    // makes code reference _CrtDbgReport and friends from the debug CRT (MSVCRTD).
    // Linking against those symbols then fails because Rust uses the release CRT.
    // Forcing cmake to build in Release mode keeps the two CRTs consistent.
    #[cfg(target_env = "msvc")]
    cmake_config.profile("Release");

    // On Linux, disable native CPU feature probing (GGML_NATIVE=ON by default).
    // GCC 12 on aarch64 detects "+fp16fml" from the native -mcpu flags and
    // reports __ARM_FEATURE_FP16_VECTOR_ARITHMETIC as defined, but the
    // always_inline fp16 NEON intrinsics (vfmaq_f16, vaddq_f16, …) then fail
    // to inline because they require the full "+fp16" target feature rather
    // than the multiply-accumulate-long variant.  Turning GGML_NATIVE off
    // leaves no architecture-specific march flags so the fp16 compile-check
    // correctly fails and the problematic code path is never compiled.
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    cmake_config.define("GGML_NATIVE", "OFF");

    // Enable CUDA backend when the "cuda" feature is active.
    #[cfg(feature = "cuda")]
    cmake_config.define("GGML_CUDA", "ON");

    let dst = cmake_config.build();

    // Tell cargo where to find the built library
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src/ggml-blas", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src/ggml-metal", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src/ggml-cuda", dst.display());
    println!("cargo:rustc-link-search=native={}/build/src", dst.display());

    // On MSVC, cmake places artifacts in a configuration subdirectory.
    // We always build in Release mode on MSVC (see above).
    #[cfg(target_env = "msvc")]
    {
        println!("cargo:rustc-link-search=native={}/build/Release", dst.display());
        println!("cargo:rustc-link-search=native={}/build/ggml/src/Release", dst.display());
        println!("cargo:rustc-link-search=native={}/build/src/Release", dst.display());
    }

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

    // On Linux with CUDA, link ggml-cuda and the CUDA runtime libraries.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
    }

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");

    // Link OpenMP (required by ggml-cpu on Linux)
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=gomp");

    // Link pthreads (Windows uses native Win32 threads, no pthread needed)
    #[cfg(not(windows))]
    println!("cargo:rustc-link-lib=pthread");

    // Rerun if llama.cpp changes
    println!("cargo:rerun-if-changed=llama.cpp/include/llama.h");
    println!("cargo:rerun-if-changed=build.rs");

    // On Windows, locate libclang.dll for bindgen if LIBCLANG_PATH is not already set
    #[cfg(windows)]
    setup_libclang_path();

    // On Linux, locate libclang.so for bindgen if LIBCLANG_PATH is not already set
    #[cfg(target_os = "linux")]
    setup_libclang_path();

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
        .allowlist_function("llama_model_size")
        .allowlist_function("llama_model_free")
        .allowlist_function("llama_model_get_vocab")
        .allowlist_function("llama_model_chat_template")
        .allowlist_function("llama_init_from_model")
        .allowlist_function("llama_free")
        .allowlist_function("llama_n_ctx")
        .allowlist_function("llama_n_batch")
        .allowlist_function("llama_get_memory")
        .allowlist_function("llama_memory_seq_add")
        .allowlist_function("llama_memory_seq_rm")
        .allowlist_function("llama_memory_seq_pos_max")
        .allowlist_function("llama_memory_clear")
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
        .allowlist_function("llama_model_n_layer")
        .allowlist_function("llama_model_n_embd")
        .allowlist_function("llama_model_n_embd_out")
        .allowlist_function("llama_max_parallel_sequences")
        .allowlist_function("llama_batch_init")
        .allowlist_function("llama_batch_free")
        .allowlist_function("llama_pooling_type")
        .allowlist_function("llama_get_embeddings_seq")
        .allowlist_function("llama_get_embeddings_ith")
        .allowlist_function("ggml_backend_load_all")
        .allowlist_function("ggml_backend_dev_count")
        .allowlist_function("ggml_backend_dev_get")
        .allowlist_function("ggml_backend_dev_type")
        .allowlist_function("ggml_backend_dev_memory")
        // Variables
        .allowlist_var("LLAMA_DEFAULT_SEED")
        .allowlist_var("LLAMA_FLASH_ATTN_TYPE_ENABLED")
        .allowlist_var("GGML_LOG_LEVEL_ERROR")
        .allowlist_var("GGML_BACKEND_DEVICE_TYPE_GPU")
        .allowlist_var("GGML_BACKEND_DEVICE_TYPE_GPU")
        .allowlist_var("GGML_TYPE_Q8_0")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}

/// On Windows, find and set LIBCLANG_PATH if not already set so bindgen can find libclang.dll.
#[cfg(windows)]
fn setup_libclang_path() {
    if std::env::var("LIBCLANG_PATH").is_ok() {
        return;
    }
    if let Some(dir) = find_libclang_dir() {
        // set_var is unsafe in Rust 1.87+ due to thread-safety concerns; build scripts
        // are single-threaded at this point so the call is safe in practice.
        unsafe { std::env::set_var("LIBCLANG_PATH", &dir) };
    }
}

/// Search common locations for the directory containing libclang.dll.
#[cfg(windows)]
fn find_libclang_dir() -> Option<PathBuf> {
    // 1. Look next to clang.exe found on PATH
    if let Ok(out) = std::process::Command::new("where.exe").arg("clang.exe").output() {
        if let Ok(s) = std::str::from_utf8(&out.stdout) {
            for line in s.lines() {
                let clang = PathBuf::from(line.trim());
                if let Some(dir) = clang.parent() {
                    if dir.join("libclang.dll").exists() {
                        return Some(dir.to_path_buf());
                    }
                }
            }
        }
    }

    // 2. Common standalone LLVM install paths
    for base in &[
        r"C:\Program Files\LLVM\bin",
        r"C:\Program Files (x86)\LLVM\bin",
    ] {
        let p = PathBuf::from(base);
        if p.join("libclang.dll").exists() {
            return Some(p);
        }
    }

    // 3. Visual Studio bundled LLVM (prefer x64, fall back to arch-neutral)
    let vs_root = PathBuf::from(r"C:\Program Files\Microsoft Visual Studio");
    if let Ok(vers) = std::fs::read_dir(&vs_root) {
        for ver in vers.flatten() {
            if let Ok(editions) = std::fs::read_dir(ver.path()) {
                for edition in editions.flatten() {
                    for arch_suffix in &[r"x64\bin", "bin"] {
                        let llvm_bin = edition
                            .path()
                            .join("VC")
                            .join("Tools")
                            .join("Llvm")
                            .join(arch_suffix);
                        if llvm_bin.join("libclang.dll").exists() {
                            return Some(llvm_bin);
                        }
                    }
                }
            }
        }
    }

    None
}

/// On Linux, find and set LIBCLANG_PATH if not already set so bindgen can find libclang.so.
#[cfg(target_os = "linux")]
fn setup_libclang_path() {
    if std::env::var("LIBCLANG_PATH").is_ok() {
        return;
    }
    if let Some(dir) = find_libclang_dir() {
        // set_var is unsafe in Rust 1.87+ due to thread-safety concerns; build scripts
        // are single-threaded at this point so the call is safe in practice.
        unsafe { std::env::set_var("LIBCLANG_PATH", &dir) };
    }
}

/// Search common locations for the directory containing libclang.so on Linux.
#[cfg(target_os = "linux")]
fn find_libclang_dir() -> Option<PathBuf> {
    // 1. Versioned LLVM installs managed by llvm.sh / apt (e.g. llvm-14 … llvm-20).
    //    Search highest version first so we prefer the newest installed toolchain.
    if let Ok(entries) = std::fs::read_dir("/usr/lib") {
        let mut llvm_dirs: Vec<PathBuf> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                if s.starts_with("llvm-") {
                    let ver: Option<u32> = s["llvm-".len()..].parse().ok();
                    ver.map(|_| e.path().join("lib"))
                } else {
                    None
                }
            })
            .filter(|p| p.is_dir())
            .collect();

        // Sort descending by version number embedded in the path name.
        llvm_dirs.sort_by(|a, b| {
            let ver = |p: &PathBuf| -> u32 {
                p.parent()
                    .and_then(|d| d.file_name())
                    .and_then(|n| n.to_str())
                    .and_then(|s| s.strip_prefix("llvm-"))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            };
            ver(b).cmp(&ver(a))
        });

        for dir in llvm_dirs {
            if has_libclang(&dir) {
                return Some(dir);
            }
        }
    }

    // 2. Multiarch lib directories (Debian/Ubuntu).
    for dir in &[
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/arm-linux-gnueabihf",
    ] {
        let p = PathBuf::from(dir);
        if has_libclang(&p) {
            return Some(p);
        }
    }

    // 3. Generic fallbacks.
    for dir in &["/usr/lib", "/usr/local/lib", "/usr/lib64"] {
        let p = PathBuf::from(dir);
        if has_libclang(&p) {
            return Some(p);
        }
    }

    None
}

/// Returns true if `dir` contains any file whose name starts with "libclang".
#[cfg(target_os = "linux")]
fn has_libclang(dir: &PathBuf) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("libclang")
            })
        })
        .unwrap_or(false)
}
