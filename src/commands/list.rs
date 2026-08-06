use anyhow::Result;
use std::path::Path;


// TODO: Add tags in table
pub fn list(all: bool) -> Result<()> {
    if all {
        // Show every model in the registry, marking which are downloaded.
        let dir = crate::models::models_dir();
        println!("{:<30} {:<40} {}", "FILE", "REPO", "STATUS");
        println!("{}", "-".repeat(78));
        for entry in crate::models::WHITELISTED_MODELS {
            let filename = find_gguf_variant(&dir, entry);
            let status = if filename.is_some() {
                "\x1b[32m✔\x1b[0m"
            } else {
                "\x1b[31m✘\x1b[0m"
            };
            println!("{:<30} {:<40} {}", filename.unwrap_or_else(|| entry.filename.to_string()), entry.repo, status);
        }
    } else {
        // Show only downloaded models.
        let dir = crate::models::models_dir();
        if !dir.exists() {
            println!("No models downloaded yet. Run `akio pull <repo>` to download one.");
            return Ok(());
        }
        let mut found = false;
        for entry in crate::models::WHITELISTED_MODELS {
            if let Some(filename) = find_gguf_variant(&dir, entry) {
                if !found {
                    println!("{:<30} {:<40}", "FILE", "REPO");
                    println!("{}", "-".repeat(72));
                    found = true;
                }
                println!("{:<30} {:<40}", filename, entry.repo);
            }
        }
        if !found {
            println!("No models downloaded yet. Run `akio pull <repo>` to download one.");
        }
    }
    Ok(())
}

/// Check if any gguf variant of the model exists on disk.
/// Returns the actual filename found (with quantization tag if present),
/// or the base filename if the base file exists directly.
fn find_gguf_variant(dir: &Path, entry: &'static crate::models::ModelEntry) -> Option<String> {
    let base = entry.filename;
    if base.is_empty() {
        return None; // multi-file models (e.g. Z-Image-Turbo)
    }

    // Check base filename first (e.g. Qwen3-0.6B.gguf)
    if dir.join(base).exists() {
        return Some(base.to_string());
    }

    // Check each quantization variant (e.g. Qwen3-0.6B-Q4_K_M.gguf)
    for tag in entry.tags {
        let variant = format!("{}-{}.gguf", base, tag);
        if dir.join(&variant).exists() {
            return Some(variant);
        }
    }

    None
}
