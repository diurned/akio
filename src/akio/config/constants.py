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


@dataclass(frozen=True)
class Color:
  """ANSI color and style codes"""
  RESET: str = "\033[0m"
  DIM: str = "\033[2m"
  BLACK: str = "\033[0;30m"
  RED: str = "\033[0;31m"
  GREEN: str = "\033[0;32m"
  YELLOW: str = "\033[0;33m"
  BLUE: str = "\033[0;34m"
  PURPLE: str = "\033[0;35m"
  CYAN: str = "\033[0;36m"
  WHITE: str = "\033[0;37m"
  BOLD_BLACK: str = "\033[1;30m"
  BOLD_RED: str = "\033[1;31m"
  BOLD_GREEN: str = "\033[1;32m"
  BOLD_YELLOW: str = "\033[1;33m"
  BOLD_BLUE: str = "\033[1;34m"
  BOLD_PURPLE: str = "\033[1;35m"
  BOLD_CYAN: str = "\033[1;36m"
  BOLD_WHITE: str = "\033[1;37m"
  UNDERLINE_BLACK: str = "\033[4;30m"
  UNDERLINE_RED: str = "\033[4;31m"
  UNDERLINE_GREEN: str = "\033[4;32m"
  UNDERLINE_YELLOW: str = "\033[4;33m"
  UNDERLINE_BLUE: str = "\033[4;34m"
  UNDERLINE_PURPLE: str = "\033[4;35m"
  UNDERLINE_CYAN: str = "\033[4;36m"
  UNDERLINE_WHITE: str = "\033[4;37m"
  BG_BLACK: str = "\033[40m"
  BG_GREY: str = "\033[48;5;235m"
  BG_RED: str = "\033[41m"
  BG_GREEN: str = "\033[42m"
  BG_YELLOW: str = "\033[43m"
  BG_BLUE: str = "\033[44m"
  BG_PURPLE: str = "\033[45m"
  BG_CYAN: str = "\033[46m"
  BG_WHITE: str = "\033[47m"
  LIGHT_BLACK: str = "\033[0;90m"
  LIGHT_RED: str = "\033[0;91m"
  LIGHT_GREEN: str = "\033[0;92m"
  LIGHT_YELLOW: str = "\033[0;93m"
  LIGHT_BLUE: str = "\033[0;94m"
  LIGHT_PURPLE: str = "\033[0;95m"
  LIGHT_CYAN: str = "\033[0;96m"
  LIGHT_WHITE: str = "\033[0;97m"
  BOLD_LIGHT_BLACK: str = "\033[1;90m"
  BOLD_LIGHT_RED: str = "\033[1;91m"
  BOLD_LIGHT_GREEN: str = "\033[1;92m"
  BOLD_LIGHT_YELLOW: str = "\033[1;93m"
  BOLD_LIGHT_BLUE: str = "\033[1;94m"
  BOLD_LIGHT_PURPLE: str = "\033[1;95m"
  BOLD_LIGHT_CYAN: str = "\033[1;96m"
  BOLD_LIGHT_WHITE: str = "\033[1;97m"
  BG_LIGHT_BLACK: str = "\033[0;100m"
  BG_LIGHT_RED: str = "\033[0;101m"
  BG_LIGHT_GREEN: str = "\033[0;102m"
  BG_LIGHT_YELLOW: str = "\033[0;103m"
  BG_LIGHT_BLUE: str = "\033[0;104m"
  BG_LIGHT_PURPLE: str = "\033[0;105m"
  BG_LIGHT_CYAN: str = "\033[0;106m"
  BG_LIGHT_WHITE: str = "\033[0;107m"
  BOLD: str = "\033[1m"
  ITALIC: str = "\033[3m"
  UNDERLINE: str = "\033[4m"
  STRIKETHROUGH: str = "\033[9m"

  @staticmethod
  def wrap(text: str, color: str) -> str:
    """
    Wrap text with a color code and reset.

    Args:
      text (str): The text content to colorize or style.
      color (str): The ANSI escape code (e.g., Color.RED, Color.BOLD_GREEN).

    Returns:
      str: The formatted text string, wrapped with the specified color/style and reset code.
    """
    return f"{color}{text}{Color.RESET}"
