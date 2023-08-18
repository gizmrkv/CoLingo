from typing import Callable, Collection, Generic, Iterable, TypeVar

import torch
import torch.nn as nn

from .runner import RunnerCallback

T = TypeVar("T")
U = TypeVar("U")


class Evaluator(RunnerCallback, Generic[T, U]):
    """
    Class for evaluating models and managing evaluation callbacks during a Runner's execution.

    Args:
        agents (Iterable[nn.Module]): List of neural network models to evaluate.
        input (Iterable[T]): List of input data for evaluation.
        games (Iterable[Callable[[T], U]]): List of evaluation functions (games) for each model and input.
        callbacks (Collection[Callable[[int, T, Iterable[U]], None]]): List of callbacks for handling evaluation results.
        intervals (Collection[int] | None, optional): List of intervals at which to perform evaluation. Defaults to None.
    """

    def __init__(
        self,
        agents: Iterable[nn.Module],
        input: Iterable[T],
        games: Iterable[Callable[[T], U]],
        callbacks: Collection[Callable[[int, T, Iterable[U]], None]],
        intervals: Collection[int] | None = None,
    ) -> None:
        super().__init__()
        self.agents = agents
        self.input = input
        self.games = games
        self.callbacks = callbacks
        self.intervals = intervals or ([1] * len(self.callbacks))

        if len(self.intervals) != len(self.callbacks):
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
            outputs = [game(input) for game in self.games]

            for flag, callback in zip(flags, self.callbacks):
                if flag:
                    callback(step, input, outputs)
