#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
from typing import Optional, Any
import json
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import ollama

from ..config.constants import Color, ConstantConfig
from ..utils.env import get_system_prompt
from ..config.settings import Settings
from ..router import Router


logger = logging.getLogger(__name__)
logging.basicConfig(filename=ConstantConfig.AKIO_LOG_FILE, level=logging.INFO)


class MCPClient:
  def __init__(self, messages: Optional[list] = None):
    self.session: Optional[ClientSession] = None
    self.exit_stack = AsyncExitStack()
    self.ollama = ollama.AsyncClient()
    self.tools = None
    self.messages = messages if messages is not None else [
      {'role': 'system', 'content': get_system_prompt()},
      {"role": "assistant", "content": "What are we breaking today?"}
    ]
    # TODO: Handle if an empty object ({}) is return.
    self.map = json.load(open(ConstantConfig.ROUTER_CONFIG_PATH)) or {"general": Settings.base_model}
    self.router = Router(self.map, cache_dir=ConstantConfig.HF_MODELS_PATH)

  async def connect_to_server(self, server_script_path: str):
    """Connect to an MCP server

    Args:
      server_script_path: Path to the server script (.py or .js)
    """
    is_python = server_script_path.endswith('.py')
    is_bun = server_script_path.endswith('.js') or server_script_path.endswith('.ts')
    if not (is_python or is_bun):
      raise ValueError("Server script must be a .py, .ts or .js file")
    # Use sys.executable to ensure we use the same Python interpreter
    # that's running the current process (important for pipx/venv installations)
    command = sys.executable if is_python else "bun"
    server_params = StdioServerParameters(
      command=command,
      args=[server_script_path],
      env=None
    )
    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
    self.stdio, self.write = stdio_transport
    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
    await self.session.initialize()
    response = await self.session.list_tools()
    self.tools = self._convert_tools(response.tools)

  def _convert_tools(self, tools) -> list | None:
    converted = []
    for tool in tools:
      schema = tool.inputSchema or {
        "type": "object",
        "properties": {},
        "required": []
      }
      ollama_tool = ollama.Tool(
        type="function",
        function=ollama.Tool.Function(
          name=tool.name,
          description=tool.description or "",
          parameters={
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
          }
        )
      )
      converted.append(ollama_tool)
    return converted

  async def auto_q(self, model: str) -> ollama.ChatResponse:
    """Automatically handle if the provided model support thinking or tool calling.

    Args:
      model (str): The model to query.

    Return:
      ollama.ChatResponse | None: The model's response, or None if an error occurs.
    """
    thinking_models: list[str] = [
      "gpt-oss", "deepseek-r1", "qwen3",
      "deepseek-v3.1", "magistral", "gpt-oss-safeguard"
    ]
    tooling_models: list[str] = [
      "athene-v2", "aya-expanse", "cogito", "command-a", "command-r",
      "command-r-plus", "command-r7b", "command-r7b-arabic", "deepseek-r1",
      "deepseek-v3.1", "devstral", "firefunction-v2", "gpt-oss",
      "gpt-oss-safeguard", "granite3-dense", "granite3-moe",
      "granite3.1-dense", "granite3.1-moe", "granite3.2", "granite3.2-vision",
      "granite3.3", "granite4", "hermes3", "llama3-groq-tool-use", "llama3.1",
      "llama3.2", "llama3.3", "llama4", "magistral", "mistral", "mistral-large",
      "mistral-nemo", "mistral-small", "mistral-small3.1", "mistral-small3.2",
      "mixtral", "nemotron", "nemotron-mini", "phi4-mini", "qwen2", "qwen2.5",
      "qwen2.5-coder", "qwen3", "qwen3-coder", "qwen3-vl", "qwq", "smollm2"
    ]
    vision_models: list[str] = [
      "bakllava", "deepseek-ocr", "gemma3", "granite3.2-vision",
      "llama3.2-vision", "llama4", "llava", "llava-llama3", "llava-phi3",
      "minicpm-v", "mistral-small3.1", "mistral-small3.2", "moondream",
      "qwen2.5vl", "qwen3-vl"
    ]
    try:
      response: ollama.ChatResponse = await self.ollama.chat(
        model=model,
        messages=self.messages,
        think=True if model.split(':')[0] in thinking_models else False,
        tools=self.tools if model.split(':')[0] in tooling_models else None,
        stream=False
      )
      return response
    except ollama.ResponseError as e:
      print(f"Ollama API error for model '{model}': {e}")
      exit(1)
    except Exception as e:
      print(f"Unexpected error during chat with model '{model}': {e}")
      exit(1)

  async def query(
    self,
    query: str,
    max_iterations: int = 10
  ) -> list[dict[str, Any]]:
    """
    Process model response and handle tool calls iteratively

    Args:
      messages (List[Dict[str, Any]]): List of messages with role/content
      max_iterations (int): Maximum number of iterations to prevent infinite loops

    Returns:
      List[Dict[str, Any]]: Updated messages list with all interactions
    """
    self.messages.append({"role": "user","content": query})
    iteration = 0
    result = self.router.route(sequence=query)
    print(
      f"\r{Color.DIM}Routes to {result.model} model.{Color.RESET}\n",
      end="",
      flush=True
    )
    while iteration < max_iterations:
      response: ollama.ChatResponse = await self.auto_q(result.model)
      logger.debug(f"\n{response}")
      message_dict = {
        'role': response.message.role,
        'content': response.message.content or ''
      }
      if response.message.tool_calls:
        message_dict['tool_calls'] = response.message.tool_calls
        self.messages.append(message_dict)
        logger.info(f"\n--- Processing {len(response.message.tool_calls)} tool call(s) ---")
        for tool in response.message.tool_calls:
          if any(tool.function.name == func.function.name for func in self.tools):
            for _, tool in enumerate(response.message.tool_calls):
              logger.info(f"Calling {tool.function.name}({tool.function.arguments})")
              print(f"{Color.BG_GREY}{Color.LIGHT_RED}Calling {tool.function.name}({tool.function.arguments}){Color.RESET}")
              output = await self.session.call_tool(
                tool.function.name,
                tool.function.arguments
              )
              logger.info(f'Function output: {output}')
              self.messages.append({
                'role': 'tool',
                'content': str(output),
                'name': tool.function.name,
                'arguments': tool.function.arguments
              })
          else:
            logger.error(f'Function {tool.function.name} not found')
      else:
        # TODO: Check if tool calling is in content.
        self.messages.append(message_dict)
        iteration += 1
        return self.messages
        break
    if iteration >= max_iterations:
      logger.warning(f"\nReached maximum iterations ({max_iterations}). Stopping auto-execution.")
    return self.messages

  async def cleanup(self):
    """Clean up resources"""
    await self.exit_stack.aclose()
