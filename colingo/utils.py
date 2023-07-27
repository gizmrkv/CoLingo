import random
from itertools import islice
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn, optim
from torchtyping import TensorType

from .core import Callback
from .logger import Logger


def fix_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
        if isinstance(m.weight_ih_l0, torch.Tensor):
            nn.init.kaiming_uniform_(m.weight_ih_l0)
        if isinstance(m.weight_hh_l0, torch.Tensor):
            nn.init.kaiming_uniform_(m.weight_hh_l0)
        if isinstance(m.bias_ih_l0, torch.Tensor):
            nn.init.zeros_(m.bias_ih_l0)
        if isinstance(m.bias_hh_l0, torch.Tensor):
            nn.init.zeros_(m.bias_hh_l0)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def random_split(dataset: torch.Tensor, propotions: list[float]) -> list[torch.Tensor]:
    indices = np.random.permutation(len(dataset))

    propotions_sum = sum(propotions)
    split_sizes = [int(r / propotions_sum * len(dataset)) for r in propotions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[torch.tensor(idx)] for idx in split_indices]

    return split_dataset


class Trainer(Callback):
    def __init__(
        self,
        game: nn.Module,
        optimizers: Iterable[optim.Optimizer],
        input: Iterable[Any],
        loss: Callable[[Any], TensorType[float]],
        max_batches: int = 1,
    ):
        self._game = game
        self._optimizers = optimizers
        self._input = input
        self._loss = loss
        self._max_batches = max_batches

    def on_update(self, step: int) -> None:
        self._game.train()
        for optimizer in self._optimizers:
            optimizer.zero_grad()

        for input in islice(self._input, self._max_batches):
            output = self._game(input)
            loss = self._loss(output)
            loss.backward(retain_graph=True)
            for optimizer in self._optimizers:
                optimizer.step()


class Evaluator(Callback):
    def __init__(
        self,
        game: nn.Module,
        input: Iterable[Any],
        loggers: Iterable[Logger] | None = None,
        run_on_begin: bool = False,
        run_on_end: bool = True,
    ):
        self._game = game
        self._input = input
        self._loggers = loggers or []
        self._run_on_begin = run_on_begin
        self._run_on_end = run_on_end

    def on_begin(self) -> None:
        if self._run_on_begin:
            self.evaluate()

    def on_update(self, step: int) -> None:
        self.evaluate()

    def on_end(self) -> None:
        if self._run_on_end:
            self.evaluate()

    def evaluate(self) -> None:
        self._game.eval()

        input = next(iter(self._input))
        with torch.no_grad():
            output = self._game(input=input)

        for logger in self._loggers:
            logger.log(output)


class StepCounter(Callback):
    def __init__(self, name: str, loggers: Iterable[Logger]) -> None:
        super().__init__()
        self._loggers = loggers
        self._name = name
        self._step = 0

    def on_update(self, step: int) -> None:
        self._step = step
        for logger in self._loggers:
            logger.log({self._name: self._step})


class ShuffleCallback(Callback):
    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self._callbacks = callbacks
        self._indices = list(range(len(self._callbacks)))

    def on_begin(self) -> None:
        for callback in self._callbacks:
            callback.on_begin()

    def on_update(self, step: int) -> None:
        random.shuffle(self._indices)
        for i in self._indices:
            self._callbacks[i].on_update(step)

    def on_end(self) -> None:
        for callback in self._callbacks:
            callback.on_end()


def shuffle(callbacks: Sequence[Callback]) -> ShuffleCallback:
    return ShuffleCallback(callbacks)


class IntervalCallback(Callback):
    def __init__(self, interval: int, callbacks: Sequence[Callback]) -> None:
        self._callbacks = callbacks
        self._interval = interval

    def on_begin(self) -> None:
        for callback in self._callbacks:
            callback.on_begin()

    def on_update(self, step: int) -> None:
        if step % self._interval == 0:
            for callback in self._callbacks:
                callback.on_update(step)

    def on_end(self) -> None:
        for callback in self._callbacks:
            callback.on_end()


def interval(interval: int, callbacks: Sequence[Callback]) -> IntervalCallback:
    return IntervalCallback(interval, callbacks)
