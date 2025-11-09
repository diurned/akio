#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import readline
import threading
import time
import logging
import json
from ..mcp.client import MCPClient
# from ..utils.env import get_system_prompt
from ...config.constants import ConstantConfig, Color


logger = logging.getLogger(__name__)
readline.parse_and_bind("tab: complete")
readline.parse_and_bind("set editing-mode vi")


def _help() -> None:
  with open(ConstantConfig.AKIO_CONFIG_FILE, 'r') as file:
    config = json.load(file)
  model = config.get('base_model')
  title = f"{Color.RED}>_{Color.RESET} Akio ({ConstantConfig.VERSION})"
  model_text = f"model:     {model}   /model to change"
  directory_text = f"directory: {os.getcwd()}"
  content_lines = [title, model_text, directory_text]
  box_width = max(len(line) for line in content_lines) + 2
  def pad(line: str) -> str:
    return f"│ {line:<{box_width - 2}} │"
  print(
    f"╭{'─' * (box_width)}╮\n"
    f"{pad(title+'                             ')}\n"
    f"{pad('')}\n"
    f"{pad(model_text)}\n"
    f"{pad(directory_text)}\n"
    f"╰{'─' * (box_width)}╯\n"
  )
  print(
    f"{Color.LIGHT_GREEN}Commands:{Color.RESET}\n"
    f"\t{Color.LIGHT_BLUE}/model{Color.RESET}: Choose which model to use\n"
    f"\t{Color.LIGHT_BLUE}/quit{Color.RESET}, {Color.LIGHT_BLUE}/exit{Color.RESET}: Exit the chat\n"
    f"\t{Color.LIGHT_BLUE}/help{Color.RESET}: Display this message\n"
  )


def start_loading(loading_active: threading.Event) -> None:
  spinner = [
    # "|", "/", "-", "\\"
    # "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
    # "⣾ ", "⣽ ", "⣻ ", "⢿ ", "⡿ ", "⣟ ", "⣯ ", "⣷ "
    # "⢄", "⢂", "⢁", "⡁", "⡈", "⡐", "⡠"
    # "█", "▓", "▒", "░"
    # "∙∙∙", "●∙∙", "∙●∙", "∙∙●"
    # "🌍", "🌎", "🌏"
    # "🙈", "🙉", "🙊"
    # "▱▱▱", "▰▱▱", "▰▰▱", "▰▰▰", "▰▰▱", "▰▱▱", "▱▱▱",
    # "☱", "☲", "☴", "☲"
    # "", ".", "..", "..."
    "🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"
  ]
  i = 0
  while loading_active.is_set():
    print(
      f"\r{Color.DIM}Thinking...{Color.RESET} {spinner[i % len(spinner)]}",
      end="",
      flush=True
    )
    time.sleep(0.1)
    i += 1
  print(f"\r{Color.DIM}Done thinking.{Color.RESET}\n", end="", flush=True)


def stop_loading(loading_active: threading.Event) -> None:
  loading_active.clear()


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
        query = input(f"{Color.BG_GREY}{Color.DIM}Ask anything {Color.LIGHT_RED}->{Color.RESET} ")
        if query.startswith('/'):
          if query.lower() in ['/quit', '/exit']:
            break
          elif query.lower() == "/model":
            print(
              "Coming soon...\n"
              "Edit the following file to change model\n"
              f"\t{ConstantConfig.AKIO_CONFIG_FILE}")
          elif query.lower() == '/help':
            _help()
            continue
        else:
          loading_active = threading.Event()
          loading_active.set()
          spinner_thread = threading.Thread(
            target=start_loading,
            args=(loading_active,)
          )
          spinner_thread.start()
          response = await mcp_client.query(query)
          stop_loading(loading_active)
          spinner_thread.join()
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
