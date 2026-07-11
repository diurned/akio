mod args;
mod commands;
mod utils;
mod ffi;
mod inference;
mod mcp;
mod models;
mod tools;
mod tui;
mod image;

use anyhow::Result;
use args::{Cli, Commands, McpAction};
use clap::Parser;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result: Result<()> = match cli.command {
        Commands::Pull { model } => commands::pull::pull(&model).await,
        Commands::Run { model, context_size, batch_size, n_gpu_layers, verbose, prompt } => {
            let prompt = if prompt.is_empty() { None } else { Some(prompt.join(" ")) };
            commands::run::run(&model, context_size, batch_size, n_gpu_layers, &verbose, prompt.as_deref()).await
        }
        Commands::Rm { model } => commands::rm::rm(&model),
        Commands::List { all } => commands::list::list(all),
        Commands::Mcp { action } => match action {
            McpAction::List => commands::mcp::list().await,
            McpAction::Add { name, command, args } => {
                commands::mcp::add(&name, &command, &args)
            }
            McpAction::Remove { name } => commands::mcp::remove(&name),
        },
        Commands::Image {
            model, prompt, height, width, num_steps, seed, output, cpu, negative_prompt, guidance_scale
        } => {
            commands::image::run(&model, prompt, height, width, num_steps, seed, output, cpu, negative_prompt, guidance_scale)
        }
        Commands::Embedding { model, inputs, n_gpu_layers, verbose } => {
            commands::embedding::embedding(&model, &inputs, n_gpu_layers, &verbose)
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
