from typing import Any, Callable, Collection, Generic, Iterable, TypeVar

import torch
import torch.nn as nn

from .abstract import Computable, Playable, Task

T = TypeVar("T")
U = TypeVar("U")


class Evaluator(Task, Generic[T, U]):
    def __init__(
        self,
        agents: Iterable[nn.Module],
        input: Iterable[T],
        game: Playable[T, U],
        metrics: Collection[Computable[T, U, Any]],
        intervals: Collection[int] | None = None,
    ) -> None:
        super().__init__()
        self.agents = agents
        self.input = input
        self.game = game
        self.metrics = metrics
        self.intervals = intervals or ([1] * len(self.metrics))

        if len(self.intervals) != len(self.metrics):
            raise ValueError(
                "The number of intervals must be the same as the number of callbacks."
            )

    def on_update(self, step: int) -> None:
        flags = [step % interval == 0 for interval in self.intervals]

        if not any(flags):
            return

        for agent in self.agents:
            agent.eval()

        input = next(iter(self.input))
        with torch.no_grad():
            output = self.game.play(input, step=step)

            for flag, metric in zip(flags, self.metrics):
                if flag:
                    metric.compute(input, output, step=step)
