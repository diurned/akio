#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from ..mcp.client import MCPClient
# from ..utils.env import get_system_prompt
from ...config.constants import ConstantConfig, Color


logger = logging.getLogger(__name__)


def _help() -> None:
  """Help function"""
  print(
    "╭─────────────────────────────────────────╮\n"
    f"│ >_ Akio ({ConstantConfig.VERSION})                     │\n"
    "│                                         │\n"
    "│ model:     qwen3:14b   /model to change │\n"
    "│ directory: ~/Developer/GitHub/akio      │\n"
    "╰─────────────────────────────────────────╯\n"
  )
  print(
    f"{Color.LIGHT_GREEN}Commands:{Color.RESET}\n"
    f"\t{Color.LIGHT_BLUE}/quit{Color.RESET}, {Color.LIGHT_BLUE}/exit{Color.RESET}: Exit the chat\n"
    f"\t{Color.LIGHT_BLUE}/help{Color.RESET}: Display this message"
  )


async def interactive_chat() -> None:
  """Initialize and run the TUI chat interface."""
  # TODO: retrieve messages from pocketbase
  # messages = [
  #   {'role': 'system', 'content': get_system_prompt()},
  #   {"role": "assistant", "content": "What are we breaking today?"}
  # ]
  mcp_client = MCPClient()
  try:
    _help()
    await mcp_client.connect_to_server(
      str(ConstantConfig.MCP_SERVER_PATH)
    )
    while True:
      try:
        query = input("Ask anything -> ")
        if query.startswith('/'):
          if query.lower() in ['/quit', '/exit']:
            break
          elif query.lower() == "/model":
            print(
              "Coming soon...\n"
              "Edit the following file to change model\n"
              "\t~/.akio/settings.json")
          elif query.lower() == '/help':
            _help()
            continue
        else:
          response = await mcp_client.query(query)
          if response and len(response) > 0:
            print(response[-1].get('content', 'No response content available.'))
          else:
            print("No response received from the assistant.")
      except KeyboardInterrupt:
        print("Use 'quit' or 'exit' to leave gracefully.", "Interrupted")
      except EOFError:
        break
      except Exception as e:
        logger.error(e)
  except Exception as e:
    logger.error(e)
    print(e)
  finally:
    await mcp_client.cleanup()
