import random
import time
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from .core import IStopper, RunnerCallback


def fix_seed(seed: int) -> None:
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


def random_split(dataset: torch.Tensor, proportions: list[float]) -> list[torch.Tensor]:
    indices = np.random.permutation(len(dataset))

    proportions_sum = sum(proportions)
    split_sizes = [int(r / proportions_sum * len(dataset)) for r in proportions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[torch.tensor(idx)] for idx in split_indices]

    return split_dataset


class EarlyStopper(RunnerCallback, IStopper):
    def __init__(self, pred: Callable[[dict[str, float]], bool]) -> None:
        self.pred = pred
        self.metrics: dict[str, float] = {}

        self._stop = False

    def __call__(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def stop(self, step: int) -> bool:
        return self._stop

    def on_update(self, step: int) -> None:
        self._stop = self.pred(self.metrics)
        self.metrics.clear()

    def on_begin(self) -> None:
        self.metrics.clear()

    def on_end(self) -> None:
        self.metrics.clear()


class DuplicateChecker(RunnerCallback):
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, keys: Iterable[str]) -> None:
        for k in keys:
            if k in self.seen:
                raise ValueError(f"Duplicate key: {k}")
            self.seen.add(k)

    def on_begin(self) -> None:
        self.seen.clear()

    def on_update(self, step: int) -> None:
        self.seen.clear()

    def on_end(self) -> None:
        self.seen.clear()


class StepCounter(RunnerCallback):
    def __init__(
        self,
        name: str,
        callbacks: Iterable[Callable[[dict[str, float]], None]],
    ) -> None:
        super().__init__()
        self.name = name
        self.callbacks = callbacks

    def on_update(self, step: int) -> None:
        for callback in self.callbacks:
            callback({self.name: step})


class Timer(RunnerCallback):
    def __init__(self, callbacks: Iterable[RunnerCallback]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def on_update(self, step: int) -> None:
        for i, callback in enumerate(self.callbacks):
            torch.cuda.synchronize()
            start = time.time()
            callback.on_update(step)
            torch.cuda.synchronize()
            end = time.time()
            print(f"{i}st time: {end - start:.3f} sec")


class Interval(RunnerCallback):
    def __init__(self, interval: int, callbacks: Iterable[RunnerCallback]) -> None:
        super().__init__()
        self.interval = interval
        self.callbacks = callbacks

    def on_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def on_update(self, step: int) -> None:
        if step % self.interval == 0:
            for callback in self.callbacks:
                callback.on_update(step)
