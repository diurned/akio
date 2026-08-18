use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiBuilder;

/// Files needed for Z-Image-Turbo (multi-file model).
const Z_IMAGE_FILES: &[&str] = &[
    "tokenizer/tokenizer.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
];

/// Split a model identifier into repo and optional quantization tag.
/// e.g. "Fastiraz/Qwen3-0.6B-GGUF:Q4_0" -> ("Fastiraz/Qwen3-0.6B-GGUF", Some("Q4_0"))
fn parse_model_name(name: &str) -> (&str, Option<&str>) {
    if let Some(pos) = name.find(':') {
        (&name[..pos], Some(&name[pos + 1..]))
    } else {
        (name, None)
    }
}

pub async fn pull(name: &str) -> Result<()> {
    let (repo, tag) = parse_model_name(name);
    let entry = crate::models::find_by_repo(repo).ok_or_else(|| {
        let list = crate::models::WHITELISTED_MODELS
            .iter()
            .map(|m| format!("  {}  ({})", m.repo, m.filename))
            .collect::<Vec<_>>()
            .join("\n");
        anyhow::anyhow!(
            "'{}' is not in the model whitelist.\n\nAvailable models:\n{}",
            name,
            list
        )
    })?;

    // Resolve the quantization tag
    let tag = tag
        .or_else(|| {
            // No tag specified — use the first available quantization
            entry.tags.first().copied()
        })
        .unwrap_or("Q4_K_M"); // fallback: most common quantization

    // Multi-file model (e.g. Z-Image-Turbo): filename is empty
    if entry.filename.is_empty() {
        return pull_multi_file(entry.repo).await;
    }

    let filename = format!("{}-{}.gguf", entry.filename, tag);
    let dest = crate::models::model_path(&filename);
    if dest.exists() {
        println!("'{}' is already downloaded at {}", filename, dest.display());
        return Ok(());
    }

    std::fs::create_dir_all(crate::models::models_dir())
        .context("failed to create models directory")?;

    println!("Pulling {}/{} ...", entry.repo, filename);

    let api = ApiBuilder::new()
        .build()
        .context("failed to initialise Hugging Face API")?;

    let cached = api
        .model(entry.repo.to_string())
        .get(&filename)
        .await
        .with_context(|| format!("failed to download {}/{}", entry.repo, filename))?;

    std::fs::copy(&cached, &dest)
        .with_context(|| format!("failed to copy model to {}", dest.display()))?;

    println!("\x1b[32m✔︎\x1b[0m Saved to {}", dest.display());
    Ok(())
}

async fn pull_multi_file(repo: &str) -> Result<()> {
    let base_dir = crate::models::model_repo_dir(repo);

    let api = ApiBuilder::new()
        .build()
        .context("failed to initialise Hugging Face API")?;
    let hf_repo = api.model(repo.to_string());

    let files = Z_IMAGE_FILES;
    let total = files.len();

    println!("Pulling {} ({} files) ...", repo, total);

    for (i, file) in files.iter().enumerate() {
        let dest = base_dir.join(file);
        if dest.exists() {
            println!("[{}/{}] {} (cached)", i + 1, total, file);
            continue;
        }

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }

        println!("[{}/{}] Downloading {} ...", i + 1, total, file);
        let cached = hf_repo
            .get(file)
            .await
            .with_context(|| format!("failed to download {}/{}", repo, file))?;

        std::fs::copy(&cached, &dest)
            .with_context(|| format!("failed to copy {} to {}", file, dest.display()))?;
    }

    println!("\x1b[32m✔︎\x1b[0m Saved to {}", base_dir.display());
    Ok(())
}
