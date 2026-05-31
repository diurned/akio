use anyhow::{bail, Result};

pub fn rm(name: &str) -> Result<()> {
    let entry = crate::models::find_by_any(name)
        .ok_or_else(|| anyhow::anyhow!("'{}' is not a known model", name))?;

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

    let path = crate::models::model_path(entry.filename);
    if !path.exists() {
        bail!("'{}' is not downloaded", entry.filename);
    }

    std::fs::remove_file(&path)?;
    println!("\x1b[31m✘\x1b[0m Removed {}", path.display());
    Ok(())
}
