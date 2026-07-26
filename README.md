<div align="center">
  <img src="./assets/logo.png" width="250">
</div>

<div align="center">
  <h1><code>akio</code></h1>
  <p>An autonomous AI agent written in Rust with in-binary inference.</p>
</div>

---

> [!WARNING]
> **Akio is under active development.**
> The project is not yet fully functional and may not work as expected out of the box.
> If you'd like to contribute, feel free to open a [pull request](https://github.com/Fastiraz/akio/pulls).
> Found a bug, vulnerability, or something unusual? Please open an [issue](https://github.com/Fastiraz/akio/issues).

---

## What is Akio?

Akio is a **plug-and-play autonomous AI agent** with **embedded model inference**, written in **Rust**. No OpenAI, no Anthropic, no Ollama — the inference runs directly inside the binary. It combines the power of large language models (LLMs) with real-world capabilities by integrating with a shell, read/write tools, a glob and websearch tools. Need more tools? Akio support MCP servers.

---

## Roadmap features

- [x] LLM inference
- [x] Default tools (shell, read, write, glob, fetch and websearch)
- [x] MCP
- [ ] API
- [ ] Orchestrator
    - [ ] Model routing
    - [ ] System prompt for each worker
    - [ ] Built-in tools for each worker
- [ ] Voice mode
    - [ ] VAD inference
    - [ ] STT inference
    - [ ] TTS inference
- [x] Image model inference
- [ ] Video model inference

---

## The Grimoire

The Grimoire is the official Akio's documentation. You can read it [here](https://fastiraz.github.io/grimoire/).

---

## Installation

### Using cargo (recommend)

```sh
cargo install --git https://github.com/Fastiraz/akio.git
```

---

## Usage

1. Download your first model

```sh
akio pull Fastiraz/Qwen3-8B-GGUF
```

2. Run the model

```sh
akio run -m Qwen3-8B-Q4_K_M.gguf
```
