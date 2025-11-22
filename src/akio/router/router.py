#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
  from dataclasses import dataclass
  import transformers
  from transformers.utils.logging import disable_progress_bar
except ModuleNotFoundError as e:
  print(
    f"Mandatory dependencies are missing: {e}"
    "Please install them with python3 -m pip install --no-cache-dir --upgrade -r requirements.txt"
  )
  exit(1)
except ImportError as e:
  print(
    "An error occurred while loading the dependencies!"
    f"Details:\n{e}"
  )
  exit(1)
except KeyboardInterrupt:
  exit(1)


@dataclass
class Result:
  id: int
  tag: str
  model: str


class Router:
  def __init__(self, map: dict | None = None, cache_dir: str | None = None):
    self.sequence: str
    self.tags: list[str] = []
    self.models: list[str] = []
    self.map: dict | None = map
    self.classifier_model: str = "facebook/bart-large-mnli"
    self.cache_dir: str | None = cache_dir
    disable_progress_bar()
    self.classifier = transformers.pipeline(
      task="zero-shot-classification",
      model=self.classifier_model,
      model_kwargs={"cache_dir": self.cache_dir},
    )


  def _set_tags_and_models(self) -> None:
    """Extracts tags and models from the provided map."""
    # TODO: optimize this code.
    if not self.tags or not self.models:
      if self.map:
        for k, v in self.map.items():
          self.tags.append(k)
          self.models.append(v)


  def route(
    self,
    sequence: str,
    map: dict | None = None
  ) -> Result:
    """
    Routes a prompt to the appropriate model based on classification.

    Args:
      sequence (str): The input string to classify and route.
      map (dict): Mapping of tags to model names.

    Returns:
      Result: The ID, tag, and model name corresponding to the highest scoring tag.

    Raises:
      ValueError: If map is empty or tag not found in map.
    """
    if map:
      self.map = map

    if not self.map:
      raise ValueError("Map cannot be empty")

    self._set_tags_and_models()

    self.sequence = sequence
    result = self.classifier(
      self.sequence,
      self.tags
    )

    tag = result["labels"][0]
    if tag not in self.map:
      raise ValueError(
        f"Tag '{tag}' not found in map. Available tags: {list(self.map.keys())}"
      )

    return Result(
      id=self.tags.index(tag),
      tag=tag,
      model=self.map[tag]
    )


def main() -> None:
  print("This is a simple example of the router's usage.")

  tags = [
    "general",
    "code",
    "math",
    "hacking"
  ]

  models = [
    "gpt-oss:20b",
    "qwen2.5-coder:14b",
    "mathstral:latest",
    "deepseek-r1:8b"
  ]

  map = {
    tags[0]: models[0],
    tags[1]: models[1],
    tags[2]: models[2],
    tags[3]: models[3]
  }

  router = Router(map, cache_dir="~/.r0uter/models")

  prompts = [
    "What's 13+37?",
    "Write the Two Sum algorithm.",
    "How to bake a cheese burger?",
    "How to brute force the hash of a password?"
  ]

  for prompt in prompts:
    result = router.route(sequence=prompt)
    print(
      f"Prompt: {prompt}\n"
      f"ID: {result.id}\n"
      f"Subject: {result.tag}\n"
      f"Model: {result.model}\n"
    )


if __name__ == "__main__":
  main()
