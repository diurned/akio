#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ...config.constants import ConstantConfig


app = typer.Typer(help="An agentic AI for red team tasks.")
console = Console()


def _serve_api(host: str = "127.0.0.1", port: int = 1337) -> None:
  """Start the FastAPI server."""
  try:
    cmd = [
      "uv", "run", "fastapi", "run", str(ConstantConfig.API_SERVER_PATH),
      f"--host={host}",
      f"--port={port}",
      "--reload"
    ]
    subprocess.run(cmd, check=True)
  except subprocess.CalledProcessError as e:
    console.print(f"[bold red]Error starting server:[/bold red] {e}")
    sys.exit(1)
  except KeyboardInterrupt:
    console.print("\n[bold yellow]Server shutdown requested by user.[/bold yellow]")
  except Exception as e:
    console.print(f"[bold red]Unexpected error:[/bold red] {e}")
    sys.exit(1)


@app.command(help="🚀 Start the FastAPI server.")
def serve(
  host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind the server to."),
  port: int = typer.Option(1337, "--port", "-p", help="Port to bind the server to.")
) -> None:
  """Start the FastAPI backend server."""
  console.print(f"[bold cyan]Starting FastAPI server[/bold cyan] on [green]{host}:{port}[/green]...")
  _serve_api(host, port)


@app.command(help="💬 Run the TUI chat loop.")
def tui() -> None:
  """Launch the Text-based User Interface (chat assistant)."""
  from ..tui.app import tui
  asyncio.run(tui())


@app.command(help="📘 Show all available commands and usage examples.")
def help() -> None:
  """Custom help display with color and categories."""
  table = Table(
    title="⚙️  CLI Commands Overview",
    title_style="bold magenta",
    header_style="bold cyan",
    show_lines=True,
  )
  table.add_column("Command", justify="right", style="bold green")
  table.add_column("Description", style="white")
  table.add_row("serve", "Start the FastAPI server (API backend).")
  table.add_row("tui", "Run the Text-based User Interface assistant.")
  table.add_row("help", "Show this help message.")
  console.print(table)

  console.print(
    Panel.fit(
      f"""
[bold white]Examples:[/bold white]
  [cyan]{sys.argv[0]} serve[/cyan]
  [cyan]{sys.argv[0]} serve --host 0.0.0.0 --port 8080[/cyan]
  [cyan]{sys.argv[0]} tui[/cyan]
""",
      title="[bold yellow]Usage Examples[/bold yellow]",
      border_style="bright_blue",
    )
  )


async def _direct_query(prompt: str) -> None:
  """Handle direct prompt input."""
  console.print(Panel.fit(f"[bold white]Prompt:[/bold white] {prompt}", title="Prompt", border_style="cyan"))
  # Example: integrate your assistant here
  try:
    from ..mcp.client import MCPClient
    mcp_client = MCPClient()
    await mcp_client.connect_to_server(
      str(ConstantConfig.MCP_SERVER_PATH)
    )
    response = await mcp_client.query(prompt)
    if response and len(response) > 0:
      ai_response = response[-1].get('content', 'No response content available.')
      print(ai_response)
    else:
      print("No response received from the assistant.")
  except ImportError:
    console.print("[yellow]No prompt handler found — implement handle_prompt() to process the input.[/yellow]")
  except KeyboardInterrupt:
    exit(1)
  except Exception as e:
    # logger.error(e)
    print(e)


def run_cli() -> None:
  """Main entry point for CLI execution."""
  if len(sys.argv) > 1 and sys.argv[1] not in ["serve", "tui", "help"]:
    asyncio.run(_direct_query(" ".join(sys.argv[1:])))
    return
  try:
    app()
  except KeyboardInterrupt:
    print("\n[!] Interrupted by user.")
  except Exception as e:
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)
