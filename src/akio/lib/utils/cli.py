#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import subprocess
import asyncio
import typer
from ...config.constants import ConstantConfig
from ..tui.app import interactive_chat


app = typer.Typer(help="An agentic AI for red team tasks.")


@app.command(help="Start an interactive chat session.")
def chat() -> None:
  asyncio.run(interactive_chat())


@app.command(help="Start the API server.")
def serve(
  host: str = typer.Option(
    "127.0.0.1", "--host", "-h", help="Host to bind the server to."
  ),
  port: int = typer.Option(
    1337, "--port", "-p", help="Port to bind the server to."
  ),
) -> None:
  """Start the API backend server."""
  print(f"Starting API server on {host}:{port}...")
  try:
    cmd = [
      "uv", "run", "fastapi", "run", str(ConstantConfig.API_SERVER_PATH),
      f"--host={host}",
      f"--port={port}",
      "--reload",
    ]
    subprocess.run(cmd, check=True)
  except subprocess.CalledProcessError as err:
    print(f"Error starting server: {err}")
    sys.exit(1)
  except KeyboardInterrupt:
    print("\nServer shutdown requested by user.")
  except Exception as err:
    print(f"Unexpected error: {err}")
    sys.exit(1)


async def _direct_query(prompt: str) -> None:
  """Handle direct prompt input."""
  print(f"\nPrompt: {prompt}\n")

  mcp_client = None
  try:
    from ..mcp.client import MCPClient

    mcp_client = MCPClient()
    await mcp_client.connect_to_server(str(ConstantConfig.MCP_SERVER_PATH))
    response = await mcp_client.query(prompt)

    if not response:
      print("No response received from the assistant.")
      return

    last_message = response[-1]
    if isinstance(last_message, dict):
      ai_response = last_message.get("content", "No response content available.")
    else:
      ai_response = getattr(last_message, "content", "No response content available.")

    print(ai_response or "No response content available from the assistant.")

  except ImportError:
    print("No prompt handler found — implement handle_prompt() to process the input.")
  except KeyboardInterrupt:
    sys.exit(1)
  except Exception as err:
    print(f"Error: {err}")
    import traceback
    traceback.print_exc()
  finally:
    if mcp_client:
      await mcp_client.cleanup()


def cli() -> None:
  try:
    if len(sys.argv) > 1 and sys.argv[1] not in ["--help", "-h", "serve", "chat"]:
      asyncio.run(_direct_query(" ".join(sys.argv[1:])))
      return
    if len(sys.argv) == 1:
      asyncio.run(interactive_chat())
    elif sys.argv[1] in ["--help", "-h"]:
      app(["--help"])
    else:
      app()
  except KeyboardInterrupt:
    print("\n[!] Interrupted by user.")
  except Exception as err:
    print(f"Error: {err}")
    sys.exit(1)
