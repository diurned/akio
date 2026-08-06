use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

pub fn rm(name: &str) -> Result<()> {
    let entry = if let Some(idx) = name.rfind(':') {
        let repo = &name[..idx];
        crate::models::find_by_repo(repo)
            .ok_or_else(|| anyhow::anyhow!("'{}' is not a known model", name))?
    } else {
        crate::models::find_by_any(name)
            .ok_or_else(|| anyhow::anyhow!("'{}' is not a known model", name))?
    };

    // Multi-file model (empty filename): remove the repo directory
    if entry.filename.is_empty() {
        let dir = crate::models::model_repo_dir(entry.repo);
        if !dir.exists() {
            bail!("'{}' is not downloaded", entry.repo);
        }
        std::fs::remove_dir_all(&dir)?;
        println!("\x1b[31m✘\x1b[0m Removed {}", dir.display());
        return Ok(());
    }

    let path = find_gguf_path(&crate::models::models_dir(), entry);
    if !path.exists() {
        bail!("'{}' is not downloaded", entry.filename);
    }

    std::fs::remove_file(&path)?;
    println!("\x1b[31m✘\x1b[0m Removed {}", path.display());
    Ok(())
}

/// Find any gguf variant of the model on disk.
/// Returns the path of the first variant found.
fn find_gguf_path(dir: &Path, entry: &crate::models::ModelEntry) -> PathBuf {
    let base = entry.filename;

    // Check base filename first (e.g. Qwen3-0.6B.gguf)
    let base_path = dir.join(base);
    if base_path.exists() {
        return base_path;
    }

    // Check each quantization variant (e.g. Qwen3-0.6B-Q4_K_M.gguf)
    for tag in entry.tags {
        let variant = format!("{}-{}.gguf", base, tag);
        let path = dir.join(&variant);
        if path.exists() {
            return path;
        }
    }

    // Fall back to base path even if it doesn't exist (for error message)
    base_path
}
