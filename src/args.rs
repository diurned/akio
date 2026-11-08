use clap::builder::{
    Styles,
    styling::{AnsiColor, Effects},
};
use clap::{Parser, Subcommand};

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default())
    .error(AnsiColor::BrightRed.on_default().effects(Effects::BOLD))
    .valid(AnsiColor::BrightGreen.on_default().effects(Effects::BOLD))
    .invalid(AnsiColor::BrightRed.on_default().effects(Effects::BOLD));

#[derive(Parser)]
#[command(
    styles = STYLES,
    name = "akio",
    about = "Local autonomous AI agent with embedded model inference.",
    long_about = "Never depends on a model provider or Google a command again.\nAkio is a plug-and-play autonomous AI agent that can assist you on anything.\n\n\
      EXAMPLES:
        akio run -m Qwen3.8-27B-Q4_0.gguf -c 8192",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start an interactive chat session with a model
    Run {
        /// Model name: filename (e.g. Qwen3.8-27B-Q4_0.gguf)
        #[arg(short = 'm')]
        model: String,
    },

    /// Serve akio API
    Serve {
        /// IP Address for the akio server (default 127.0.0.1)
        #[arg(short = 'H', default_value = "127.0.0.1")]
        host: String,

        /// Port for the akio server (default 1337)
        #[arg(short = 'p', default_value_t = 1337)]
        port: u16,
    },
}
