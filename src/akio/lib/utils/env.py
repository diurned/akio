#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple
import os
import hashlib
from ..tools.rag import RAG
from ...config.constants import ConstantConfig, Color


def env_info() -> Tuple[str,str]:
  """
  Returns the operating system name and the default shell.

  Returns:
    A tuple containing the OS name and the default shell.
  """
  os_name = __import__('platform').system()
  sh = os.environ.get('SHELL', 'bash')
  return os_name, sh


def get_system_prompt(file_path: str = ConstantConfig.SYSTEM_PROMPT_PATH) -> str:
  """
  Returns the system prompt for the AI agent.

  Args:
    file_path (str): The system prompt file path.

  Returns:
    str: A string containing the content of the system prompt file.
  """
  with open(file_path, 'r', encoding='utf8') as file:
    return file.read()


def load_env(env_path: str = ".env") -> None:
  """
  Load environment variables from the `.env` file.

  Args:
    env_path (str): The path to the `.env` file.
  """
  with open(env_path, 'r', encoding='utf8') as file:
    for line in file:
      line = line.strip()
      if line and not line.startswith('#'):
        if '=' in line:
          key, value = line.split('=', 1)
          key = key.strip()
          value = value.strip()
          if (value.startswith('"') and value.endswith('"')) or \
             (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
          os.environ[key] = value


def create_vdb_if_needed() -> None:
  """
  Create the vector database if it does not already exist.
  """
  rag = RAG(str(ConstantConfig.VECTOR_DB_PATH))
  if not rag.collection.count():
    print(f"{Color.DIM}No data found in vector DB. Indexing documents...{Color.RESET}")
    rag.load(ConstantConfig.DATASETS_PATH)
    rag.chunk()
    rag.vector_store()
    print(f"{Color.DIM}Vector store ready.{Color.RESET}")
  else:
    print(f"{Color.DIM}Loaded existing vector DB with {rag.collection.count()} entries.{Color.RESET}")


def hash_folder(path: str) -> str:
  """
  Return a hash representing the current state of a folder.

  Args:
    path (str): The path to the folder to check.

  Returns:
    str: The folder's hash.

  Raises:
    ValueError: If path is not a directory.
  """
  if not os.path.isdir(path):
    raise ValueError(f"The path {path} is not a directory.")
  hasher = hashlib.sha256()
  for root, dirs, files in os.walk(path):
    for name in sorted(files):
      filepath = os.path.join(root, name)
      try:
        stat = os.stat(filepath)
      except FileNotFoundError:
        continue
      hasher.update(name.encode())
      hasher.update(str(stat.st_mtime).encode())
      hasher.update(str(stat.st_size).encode())
  return hasher.hexdigest()
