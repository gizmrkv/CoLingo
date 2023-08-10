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
        game: Callable[[T], U],
        callbacks: Iterable[Callable[[U], None]],
    ) -> None:
        super().__init__()
        self.agents = agents
        self.input = input
        self.game = game
        self.callbacks = callbacks

    def on_update(self, step: int) -> None:
        for agent in self.agents:
            agent.eval()

        input = next(iter(self.input))
        with torch.no_grad():
            output = self.game(input)
            for callback in self.callbacks:
                callback(output)

    def on_begin(self) -> None:
        pass

    def on_end(self) -> None:
        pass
