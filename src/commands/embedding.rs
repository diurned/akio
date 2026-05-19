use anyhow::{bail, Result};

pub fn embedding(model: &str, inputs: &[String], n_gpu_layers: i32, verbose: &str) -> Result<()> {
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

    crate::inference::log::set_min_level(crate::utils::log::parse_log_level(verbose));
    let embeddings = crate::inference::embedding::run_embedding(path_str, inputs, n_gpu_layers)?;

    let output: Vec<serde_json::Value> = embeddings
        .iter()
        .enumerate()
        .map(|(i, embd)| {
            serde_json::json!({
                "index": i,
                "embedding": embd,
            })
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
