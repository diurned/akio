mod args;
mod commands;
mod ffi;
mod inference;
mod mcp;
mod models;
mod tools;
mod tui;

use anyhow::Result;
use args::{Cli, Commands, McpAction};
use clap::Parser;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result: Result<()> = match cli.command {
        Commands::Pull { repo } => commands::pull::pull(&repo).await,
        Commands::Run { model, context_size, n_gpu_layers } => {
            commands::run::run(&model, context_size, n_gpu_layers).await
        }
        Commands::Rm { repo } => commands::rm::rm(&repo),
        Commands::List { all } => commands::list::list(all),
        Commands::Mcp { action } => match action {
            McpAction::List => commands::mcp::list().await,
            McpAction::Add { name, command, args } => {
                commands::mcp::add(&name, &command, &args)
            }
            McpAction::Remove { name } => commands::mcp::remove(&name),
        },
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
