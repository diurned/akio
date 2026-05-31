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

        /// Set log verbosity (none, debug, info, warn, error)
        #[arg(long, default_value = "error", value_parser = ["none", "debug", "info", "warn", "error"])]
        verbose: String,

        /// Optional prompt for non-interactive mode
        // #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,
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
        #[arg(long = "all", short = 'a')]
        all: bool,
    },

    /// Generate an image from a text prompt
    #[clap(alias="img")]
    Image {
        /// Model to use
        #[arg(short = 'm', long)]
        model: String,

        /// Text prompt describing the image to generate
        #[arg(long)]
        prompt: String,

        /// The height in pixels of the generated image
        #[arg(long)]
        height: Option<usize>,

        /// The width in pixels of the generated image
        #[arg(long)]
        width: Option<usize>,

        /// Number of inference steps
        #[arg(long)]
        num_steps: Option<usize>,

        /// Random seed for reproducible output
        #[arg(long)]
        seed: Option<u64>,

        /// Output image filename
        #[arg(long)]
        output: Option<String>,

        /// Run on CPU rather than GPU
        #[arg(long)]
        cpu: bool,

        /// Negative prompt to guide generation away from
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Classifier-free guidance scale
        #[arg(long)]
        guidance_scale: Option<f64>,
    },

    /// Generate text embeddings from input texts
    #[clap(alias="embed")]
    Embedding {
        /// Path to the GGUF embedding model file
        #[arg(short = 'm')]
        model: String,

        /// Input texts to embed
        #[arg(required = true)]
        inputs: Vec<String>,

        /// Number of layers to offload to GPU
        #[arg(long = "ngl", default_value_t = 99)]
        n_gpu_layers: i32,

        /// Set log verbosity (none, debug, info, warn, error)
        #[arg(long, default_value = "error", value_parser = ["none", "debug", "info", "warn", "error"])]
        verbose: String,
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
    #[clap(alias="ls")]
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
    #[clap(alias="rm")]
    Remove {
        /// Name of the MCP server to remove
        name: String,
    },
}
