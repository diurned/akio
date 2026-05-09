use anyhow::{bail, Result};

pub fn rm(repo: &str) -> Result<()> {
    let entry = crate::models::find_by_repo(repo)
        .ok_or_else(|| anyhow::anyhow!("'{}' is not a known model repository", repo))?;

    let path = crate::models::model_path(entry.filename);
    if !path.exists() {
        bail!("'{}' is not downloaded", entry.filename);
    }

    std::fs::remove_file(&path)?;
    println!("\x1b[31m✘\x1b[0m Removed {}", path.display());
    Ok(())
}
