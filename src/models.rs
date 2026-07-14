use std::path::PathBuf;


pub struct ModelEntry {
    pub repo: &'static str,
    pub filename: &'static str,
}

pub const WHITELISTED_MODELS: &[ModelEntry] = &[
    ModelEntry {
        repo: "Fastiraz/Qwen3-0.6B-GGUF",
        filename: "Qwen3-0.6B-Q4_0.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-1.7B-GGUF",
        filename: "Qwen3-1.7B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-4B-GGUF",
        filename: "Qwen3-4B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-8B-GGUF",
        filename: "Qwen3-8B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-14B-GGUF",
        filename: "Qwen3-14B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-32B-GGUF",
        filename: "Qwen3-32B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3.5-9B-GGUF",
        filename: "Qwen3.5-9B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-Embedding-0.6B-GGUF",
        filename: "Qwen3-Embedding-0.6B-Q8_0.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-Embedding-4B-GGUF",
        filename: "Qwen3-Embedding-4B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-Embedding-8B-GGUF",
        filename: "Qwen3-Embedding-8B-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Ornith-1.0-9B-GGUF",
        filename: "ornith-1.0-9b-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Fastiraz/Ornith-1.0-35B-GGUF",
        filename: "ornith-1.0-35b-Q4_K_M.gguf",
    },
    ModelEntry {
        repo: "Tongyi-MAI/Z-Image-Turbo",
        filename: "",
    },
];

pub fn find_by_repo(repo: &str) -> Option<&'static ModelEntry> {
    WHITELISTED_MODELS.iter().find(|m| m.repo == repo)
}

pub fn find_by_filename(filename: &str) -> Option<&'static ModelEntry> {
    WHITELISTED_MODELS.iter().find(|m| m.filename == filename)
}

/// Resolve a model identifier that can be either a HuggingFace repo path or a filename.
/// Returns the corresponding ModelEntry if found.
pub fn find_by_any(name: &str) -> Option<&'static ModelEntry> {
    find_by_repo(name).or_else(|| find_by_filename(name))
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

/// Directory for multi-file models stored by repo name (e.g. "Tongyi-MAI/Z-Image-Turbo").
pub fn model_repo_dir(repo: &str) -> PathBuf {
    models_dir().join(repo)
}


pub fn resolve_model(name: &str) -> PathBuf {
    // If it's an absolute/relative path that exists, use it directly
    let p = PathBuf::from(name);
    if p.exists() {
        return p;
    }

    // If it looks like a HuggingFace repo (contains '/'), resolve to filename
    if let Some(entry) = find_by_repo(name) {
        if !entry.filename.is_empty() {
            return model_path(entry.filename);
        }
        return model_repo_dir(entry.repo);
    }

    // Otherwise treat as a filename
    model_path(name)
}
