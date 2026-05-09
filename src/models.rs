use std::path::PathBuf;


pub struct ModelEntry {
    pub repo: &'static str,
    pub filename: &'static str,
}

pub const WHITELISTED_MODELS: &[ModelEntry] = &[
    ModelEntry {
        repo: "ggml-org/Qwen3-0.6B-GGUF",
        filename: "Qwen3-0.6B-Q4_0.gguf",
    },
    ModelEntry {
        repo: "ggml-org/gpt-oss-20b-GGUF",
        filename: "gpt-oss-20b-Q4_0.gguf",
    },
    ModelEntry {
        repo: "ggml-org/Qwen3-8B-GGUF",
        filename: "Qwen3-8B-Q4_K_M.gguf",
    },
];

pub fn find_by_repo(repo: &str) -> Option<&'static ModelEntry> {
    WHITELISTED_MODELS.iter().find(|m| m.repo == repo)
}


pub fn models_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("cannot determine home directory");
    PathBuf::from(home).join(".akio").join("models")
}


pub fn model_path(filename: &str) -> PathBuf {
    models_dir().join(filename)
}


pub fn resolve_model(name: &str) -> PathBuf {
    let p = PathBuf::from(name);
    if p.exists() {
        p
    } else {
        model_path(name)
    }
}
