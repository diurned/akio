use clap::{Parser, Subcommand};
use clap::builder::{
  styling::{AnsiColor, Effects},
  Styles,
};

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
    long_about = "Never depends on a model provider or Google a command again.\nAkio is a plug-and-play autonomous AI agent that can assist you.\n\n\
      EXAMPLES:
        akio pull ggml-org/Qwen3-0.6B-GGUF
        akio run -m Qwen3-0.6B-Q4_0.gguf -c 8192
        akio rm ggml-org/Qwen3-0.6B-GGUF
        akio list
        akio list --all
        akio mcp list
        akio mcp add --name browser-use --command uvx --args \"uvx\" \"run\" \"browser-use/index.js\"
        akio mcp remove --name browser-use",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Download a model from a Hugging Face repository
    Pull {
        /// Hugging Face repository (e.g. ggml-org/Qwen3-0.6B-GGUF)
        repo: String,
    },

    /// Start an interactive chat session with a model
    Run {
        /// Path to the GGUF model file
        #[arg(short = 'm')]
        model: String,

        /// Context window size in tokens
        #[arg(short = 'c', default_value_t = 8192)]
        context_size: u32,

        /// Number of layers to offload to GPU
        #[arg(long = "ngl", default_value_t = 99)]
        n_gpu_layers: i32,
    },

    /// Remove a previously downloaded model
    Rm {
        /// Hugging Face repository to remove (e.g. ggml-org/Qwen3-0.6B-GGUF)
        repo: String,
    },

    /// List downloaded models
    #[clap(alias="ls")]
    List {
        /// Show all available GGUF files, not just the repository names
        #[arg(long)]
        all: bool,
    },

    /// Manage MCP servers
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
}

#[derive(Subcommand)]
pub enum McpAction {
    /// List registered MCP servers and their tools
    List,

    /// Add an MCP server
    Add {
        /// Friendly name for this server (e.g. "browser-use")
        #[arg(long)]
        name: String,

        /// Command to spawn the server (e.g. "uvx", "node", "npx")
        #[arg(long)]
        command: String,

        /// Arguments passed to the command
        #[arg(long, num_args = 1.., allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Remove a registered MCP server
    Remove {
        /// Name of the MCP server to remove
        name: String,
    },
}
