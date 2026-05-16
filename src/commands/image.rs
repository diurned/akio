use anyhow::{bail, Result};

pub fn run(
    model: &str,
    prompt: String,
    height: Option<usize>,
    width: Option<usize>,
    num_steps: Option<usize>,
    seed: Option<u64>,
    output: Option<String>,
    cpu: bool,
    negative_prompt: String,
    guidance_scale: Option<f64>,
) -> Result<()> {
    match model {
        "turbo" => {
            let args = crate::image::z::Args {
                prompt,
                negative_prompt,
                cpu,
                height: height.unwrap_or(1024),
                width: width.unwrap_or(1024),
                num_steps,
                guidance_scale: guidance_scale.unwrap_or(5.0),
                seed,
                model: crate::image::z::Model::Turbo,
                model_path: None,
                output: output.unwrap_or_else(|| "z_image_output.png".to_string()),
            };
            crate::image::z::run(args)
        }
        _ => bail!(
            "unknown model '{}'. Available models: schnell, dev, turbo",
            model
        ),
    }
}
