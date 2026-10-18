use std::env;
use std::path::PathBuf;
use std::process::Command;

fn link(dst: PathBuf, cuda: bool) {
    println!("cargo:warning=llama.cpp build output: {}", dst.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=llama.cpp");

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/src", dst.display());
    println!(
        "cargo:rustc-link-search=native={}/build/ggml/src",
        dst.display()
    );

    if cuda {
        println!(
            "cargo:rustc-link-search=native={}/build/ggml/src/ggml-cuda",
            dst.display()
        );
    }

    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    if cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
    }

    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
}

fn bind() {
    let bindings = bindgen::Builder::default()
        .header("llama.cpp/include/llama.h")
        // .clang_arg("-Illama.cpp/include")
        .clang_arg("-Illama.cpp/ggml/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    // NOTE: On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU. To disable the Metal build at compile time use the -DGGML_METAL=OFF cmake option.
    let mut config = cmake::Config::new("llama.cpp");
    config
        .build_target("llama")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .profile("Release");

    let cuda_available = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--count=1")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if cuda_available {
        config.define("GGML_CUDA", "ON");
    }

    let dst = config.build();

    link(dst, cuda_available);
    bind();
}
