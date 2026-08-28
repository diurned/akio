use std::path::PathBuf;

pub struct ModelEntry {
    pub repo: &'static str,
    pub tags: &'static [&'static str],
    pub filename: &'static str,
}

pub const WHITELISTED_MODELS: &[ModelEntry] = &[
    ModelEntry {
        repo: "Fastiraz/Qwen3-0.6B-GGUF",
        tags: &["Q4_0", "Q8_0", "f16"],
        filename: "Qwen3-0.6B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-1.7B-GGUF",
        tags: &["Q4_K_M", "Q8_0", "f16"],
        filename: "Qwen3-1.7B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-4B-GGUF",
        tags: &["Q4_K_M", "Q8_0", "f16"],
        filename: "Qwen3-4B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-8B-GGUF",
        tags: &["Q4_K_M", "Q8_0", "f16"],
        filename: "Qwen3-8B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-14B-GGUF",
        tags: &["Q4_K_M", "Q8_0", "f16"],
        filename: "Qwen3-14B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-32B-GGUF",
        tags: &["Q4_K_M", "Q8_0"],
        filename: "Qwen3-32B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3.5-9B-GGUF",
        tags: &[
            "Q4_K_M",
            "BF16",
            "IQ4_NL",
            "IQ4_XS",
            "Q3_K_M",
            "Q3_K_S",
            "Q4_0",
            "Q4_1",
            "Q4_K_S",
            "Q5_K_M",
            "Q5_K_S",
            "Q6_K",
            "Q8_0",
            "UD-IQ2_M",
            "UD-IQ2_XXS",
            "UD-IQ3_XXS",
            "UD-Q2_K_XL",
            "UD-Q3_K_XL",
            "UD-Q4_K_XL",
            "UD-Q5_K_XL",
            "UD-Q6_K_XL",
            "UD-Q8_K_XL",
        ],
        filename: "Qwen3.5-9B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-Embedding-0.6B-GGUF",
        tags: &["Q8_0", "f16"],
        filename: "Qwen3-Embedding-0.6B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-Embedding-4B-GGUF",
        tags: &["Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0", "f16"],
        filename: "Qwen3-Embedding-4B",
    },
    ModelEntry {
        repo: "Fastiraz/Qwen3-Embedding-8B-GGUF",
        tags: &["Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0", "f16"],
        filename: "Qwen3-Embedding-8B",
    },
    ModelEntry {
        repo: "Fastiraz/Ornith-1.0-9B-GGUF",
        tags: &["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "bf16"],
        filename: "ornith-1.0-9b",
    },
    ModelEntry {
        repo: "Fastiraz/Ornith-1.0-35B-GGUF",
        tags: &["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "bf16"],
        filename: "ornith-1.0-35b",
    },
    ModelEntry {
        repo: "Fastiraz/Ornith-1.5-9B-GGUF",
        tags: &["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"],
        filename: "Ornith-1.5-9B",
    },
    ModelEntry {
        repo: "Fastiraz/Ornith-1.5-35B-A3B-GGUF",
        tags: &["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"],
        filename: "Ornith-1.5-35B",
    },
    ModelEntry {
        repo: "Fastiraz/Ornith-1.5-397B-GGUF",
        tags: &["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        filename: "Ornith-1.5-397B",
    },
    ModelEntry {
        repo: "Fastiraz/DeepSeek-R1-1.5B-GGUF",
        tags: &[
            "Q4_K_M",
            "BF16",
            "Q2_K",
            "Q2_K_L",
            "Q3_K_M",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "UD-IQ1_M",
            "UD-IQ1_S",
            "UD-IQ2_M",
            "UD-IQ2_XXS",
            "UD-IQ3_XXS",
            "UD-IQ4_XS",
            "UD-Q2_K_XL",
            "UD-Q3_K_XL",
            "UD-Q4_K_XL",
        ],
        filename: "DeepSeek-R1-1.5B",
    },
    ModelEntry {
        repo: "Fastiraz/DeepSeek-R1-7B-GGUF",
        tags: &[
            "Q4_K_M", "F16", "Q2_K", "Q2_K_L", "Q3_K_M", "Q5_K_M", "Q6_K", "Q8_0",
        ],
        filename: "DeepSeek-R1-7B",
    },
    ModelEntry {
        repo: "Fastiraz/DeepSeek-R1-14B-GGUF",
        tags: &[
            "Q4_K_M", "F16", "Q2_K", "Q2_K_L", "Q3_K_M", "Q5_K_M", "Q6_K", "Q8_0",
        ],
        filename: "DeepSeek-R1-14B",
    },
    ModelEntry {
        repo: "Fastiraz/DeepSeek-R1-32B-GGUF",
        tags: &[
            "Q4_K_M", "F16", "Q2_K", "Q2_K_L", "Q3_K_M", "Q5_K_M", "Q6_K", "Q8_0",
        ],
        filename: "DeepSeek-R1-32B",
    },
    ModelEntry {
        // TODO: Add MTP support with Qwythos-9B-v2-MTP
        repo: "Fastiraz/Qwythos-9B-v2-GGUF",
        tags: &["Q4_K_M", "BF16", "Q5_K_M", "Q6_K", "Q8_0"],
        filename: "Qwythos-9B-v2",
    },
    ModelEntry {
        // TODO: Add MTP support with Qwythos-9B-Claude-Mythos-5-1M-MTP
        repo: "Fastiraz/Qwythos-9B-v1-GGUF",
        tags: &["BF16", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        filename: "Qwythos-9B-Claude-Mythos-5-1M",
    },
    ModelEntry {
        // TODO: Add MTP support with Qwythos-27B-MTP
        repo: "Fastiraz/Qwythos-27B-v1-GGUF",
        tags: &["BF16", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        filename: "Qwythos-27B",
    },
    ModelEntry {
        repo: "Fastiraz/Z-Image-Turbo",
        tags: &[],
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
    let mut repo: Vec<_> = if name.contains(":") {
        name.split(":").collect()
    } else {
        vec![name]
    };
    if let Some(entry) = find_by_repo(repo[0]) {
        if repo.len() < 2 {
            repo.push(entry.tags[0]);
        }
        if !entry.filename.is_empty() {
            return model_path(format!("{}-{}.gguf", entry.filename, repo[1]).as_str());
        }
        return model_repo_dir(entry.repo);
    }

    // Otherwise treat as a filename
    model_path(name)
}
