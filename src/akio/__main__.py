#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from .config.init import init
init()
from .utils.env import create_vdb_if_needed
from .cli import cli
from .config.constants import ConstantConfig


def main() -> None:
  create_vdb_if_needed()
  logging.basicConfig(
    filename=ConstantConfig.AKIO_LOG_FILE,
    level=logging.INFO,
    # format='[%(asctime)s] %(levelname)s - %(message)s',
  )
  cli()


if __name__ == "__main__":
  main()
