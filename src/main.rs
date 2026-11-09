mod api;
mod args;
mod ffi;
mod inference;

use args::{Cli, Commands};
use clap::Parser;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { model } => {
            let mut llama_cpp = inference::llm::LlamaCpp::new();
            llama_cpp.load(model);
            llama_cpp.interactive();
        }
        Commands::Serve { host, port, model } => api::chat::chat(host, port, model).await,
    };
}
