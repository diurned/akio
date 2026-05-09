use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiBuilder;

pub async fn pull(repo: &str) -> Result<()> {
    let entry = crate::models::find_by_repo(repo).ok_or_else(|| {
        let list = crate::models::WHITELISTED_MODELS
            .iter()
            .map(|m| format!("  {}", m.repo))
            .collect::<Vec<_>>()
            .join("\n");
        anyhow::anyhow!("'{}' is not in the model whitelist.\n\nAvailable models:\n{}", repo, list)
    })?;

    let dest = crate::models::model_path(entry.filename);
    if dest.exists() {
        println!("'{}' is already downloaded at {}", entry.filename, dest.display());
        return Ok(());
    }

    std::fs::create_dir_all(crate::models::models_dir())
        .context("failed to create models directory")?;

    println!("Pulling {}/{} ...", entry.repo, entry.filename);

    let api = ApiBuilder::new()
        .build()
        .context("failed to initialise Hugging Face API")?;

    let cached = api
        .model(entry.repo.to_string())
        .get(entry.filename)
        .await
        .with_context(|| format!("failed to download {}/{}", entry.repo, entry.filename))?;

    std::fs::copy(&cached, &dest)
        .with_context(|| format!("failed to copy model to {}", dest.display()))?;

    println!("\x1b[32m✔︎\x1b[0m Saved to {}", dest.display());
    Ok(())
}
