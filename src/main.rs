mod args;
mod ffi;
mod inference;

use args::{Cli, Commands};
use clap::Parser;

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { model } => inference::llm::serve(model.to_string()),
    };
}
