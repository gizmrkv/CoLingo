from typing import Callable, Generic, Iterable, TypeVar

import torch
import torch.nn as nn

from .runner import RunnerCallback

T = TypeVar("T")
U = TypeVar("U")


class Evaluator(RunnerCallback, Generic[T, U]):
    def __init__(
        self,
        agents: Iterable[nn.Module],
        input: Iterable[T],
        games: Iterable[Callable[[T], U]],
        callbacks: Iterable[Callable[[int, T, Iterable[U]], None]],
    ) -> None:
        super().__init__()
        self.agents = agents
        self.input = input
        self.games = games
        self.callbacks = callbacks

    def on_update(self, step: int) -> None:
        for agent in self.agents:
            agent.eval()

        input = next(iter(self.input))
        with torch.no_grad():
            outputs = [game(input) for game in self.games]
            for callback in self.callbacks:
                callback(step, input, outputs)
