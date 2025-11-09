#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from dataclasses import dataclass


__version__ = "0.0.2-dev"


@dataclass
class ConstantConfig:
  """Constant parameters information"""
  VERSION: str = __version__
  NAME: str = "akio"
  LANDING: str = "https://0xcat.io"
  DOCUMENTATION: str = "https://docs.0xcat.io"

  # OS Dir full root path of akio project
  SRC_ROOT_PATH_OBJ: Path = Path(__file__).parent.parent.parent.resolve()

  # Path to the EULA docs
  EULA_PATH: Path = SRC_ROOT_PATH_OBJ / f"{NAME}/utils/docs/eula.md"

  # Akio config directory
  AKIO_CONFIG_PATH: Path = Path().home() / f".{NAME}"

  # Akio config file
  AKIO_CONFIG_FILE: Path = AKIO_CONFIG_PATH / "settings.json"

  # Akio log file
  AKIO_LOG_FILE: Path = AKIO_CONFIG_PATH / f"{NAME}.log"

  # Install mode, check if Akio has been git cloned or installed using pip package
  GIT_SOURCE_INSTALLATION: bool = (SRC_ROOT_PATH_OBJ / '.git').is_dir()
  PIP_INSTALLED: bool = SRC_ROOT_PATH_OBJ.name == "site-packages"
  PIPX_INSTALLED: bool = "/pipx/venvs/" in SRC_ROOT_PATH_OBJ.as_posix()
  UV_INSTALLED: bool = "/uv/tools/" in SRC_ROOT_PATH_OBJ.as_posix()

  GITHUB_REPO: str = f"Fastiraz/{NAME}"

  # Vector database path
  VECTOR_DB_PATH: str = AKIO_CONFIG_PATH / "vectordb"

  # Datasets path
  DATASETS_PATH: str = AKIO_CONFIG_PATH / "datasets"

  # LLM system prompt path
  SYSTEM_PROMPT_PATH: str = AKIO_CONFIG_PATH / "knowledge/prompt.txt"

  # MCP server path
  MCP_SERVER_PATH: str = SRC_ROOT_PATH_OBJ / f"{NAME}/lib/mcp/server.py"

  # API server path
  API_SERVER_PATH: str = SRC_ROOT_PATH_OBJ / f"{NAME}/lib/api/routes.py"


@dataclass
class Color:
  """Colors constant"""
  RESET: str = "\033[0m"
  RED: str = "\033[31m"
  GREEN: str = "\033[32m"
  BLUE: str = "\033[34m"
  LIGHT_RED: str = "\033[91m"
  LIGHT_GREEN: str = "\033[92m"
  LIGHT_BLUE: str = "\033[94m"
  MAGENTA: str = "\033[35m"
  YELLOW: str = "\033[33m"
  CYAN: str = "\033[36m"
