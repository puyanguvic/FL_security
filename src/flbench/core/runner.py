from __future__ import annotations

from typing import Any


class SimRunner:
    """Thin wrapper around an NVFLARE recipe execution."""

    def __init__(self, recipe: Any, env: Any):
        self.recipe = recipe
        self.env = env

    def run(self) -> Any:
        return self.recipe.execute(self.env)
