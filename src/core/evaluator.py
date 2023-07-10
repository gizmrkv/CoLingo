from typing import Any, Callable, Iterable

import torch
from torch import nn

from ..logger import Logger
from .callback import Callback


class Evaluator(Callback):
    def __init__(
        self,
        game: nn.Module,
        input: Iterable,
        metrics: Callable[[list], dict[str, float | int]],
        logger: Logger | Iterable[Logger],
        run_on_begin: bool = True,
        run_on_end: bool = True,
    ):
        self._game = game
        self._input = input
        self._metrics = metrics
        self._loggers = [logger] if isinstance(logger, Logger) else logger
        self._run_on_begin = run_on_begin
        self._run_on_end = run_on_end

    def on_begin(self):
        if self._run_on_begin:
            self.evaluate()

    def on_end(self):
        if self._run_on_end:
            self.evaluate()

    def on_update(self, step: int):
        self.evaluate()

    def evaluate(self):
        self._game.eval()

        outputs = []
        for input in self._input:
            with torch.no_grad():
                output = self._game(input=input)
                outputs.append(output)

        metrics = self._metrics(outputs)

        for logger in self._loggers:
            logger.log(metrics)
