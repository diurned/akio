use anyhow::{bail, Result};

pub async fn run(model: &str, context_size: u32, n_gpu_layers: i32) -> Result<()> {
    let path = crate::models::resolve_model(model);
    if !path.exists() {
        bail!(
            "model file not found: {}\n\nRun `akio pull <repo>` to download a model.",
            path.display()
        );
    }
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path contains non-UTF-8 characters"))?;

    crate::utils::ggml::check_memory(path_str, context_size)
        .map_err(|e| anyhow::anyhow!(e))?;

    crate::inference::run_chat(path_str, context_size, n_gpu_layers).await
}
