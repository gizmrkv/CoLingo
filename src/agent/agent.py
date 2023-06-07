from abc import ABC, abstractmethod
from typing import Any

import torch as th


class Agent(ABC, th.nn.Module):
    @abstractmethod
    def input(self, game_name: str | None = None, **inputs) -> Any:
        raise NotImplementedError

    def message(self, hidden, game_name: str | None = None) -> Any:
        raise NotImplementedError
