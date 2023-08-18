import random
import time
from typing import Callable, Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from .core import EarlyStopper, RunnerCallback


def fix_seed(seed: int) -> None:
    """
    Fix random seed for reproducibility of random operations.

    Args:
        seed (int): Seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(m: nn.Module) -> None:
    """
    Initialize the weights of a neural network module.

    Args:
        m (nn.Module): The neural network module to initialize.
    """

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


def random_split(dataset: TensorType, proportions: Iterable[float]) -> List[TensorType]:
    """
    Randomly split a dataset into multiple subsets according to given proportions.

    Args:
        dataset (TensorType): The dataset to split.
        proportions (Iterable[float]): Proportions for splitting the dataset.

    Returns:
        List[TensorType]: List of subsets of the dataset.
    """
    indices = np.random.permutation(len(dataset))

    proportions_sum = sum(proportions)
    split_sizes = [int(r / proportions_sum * len(dataset)) for r in proportions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[torch.tensor(idx)] for idx in split_indices]

    return split_dataset


class MetricsEarlyStopper(RunnerCallback, EarlyStopper):
    """
    Callback for early stopping based on user-defined metrics conditions.

    Args:
        pred (Callable[[Dict[str, float]], bool]): Prediction function for early stopping.
    """

    def __init__(self, pred: Callable[[Dict[str, float]], bool]) -> None:
        self.pred = pred
        self.metrics: Dict[str, float] = {}

        self._stop = False

    def __call__(self, metrics: Dict[str, float]) -> None:
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
    """
    Callback to check for duplicate keys during execution.

    This callback helps prevent the usage of duplicate keys within the same execution run.

    Raises:
        ValueError: If a duplicate key is detected.
    """

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
    """
    Callback to count and report step numbers to other callbacks.

    Args:
        name (str): The name of the step counter.
        callbacks (Iterable[Callable[[Dict[str, float]], None]]): List of callback functions to notify with the step count.
    """

    def __init__(
        self,
        name: str,
        callbacks: Iterable[Callable[[Dict[str, float]], None]],
    ) -> None:
        super().__init__()
        self.name = name
        self.callbacks = callbacks

    def on_update(self, step: int) -> None:
        for callback in self.callbacks:
            callback({self.name: step})


class Timer(RunnerCallback):
    """
    Callback to measure execution time of other callbacks.

    Args:
        callbacks (Iterable[RunnerCallback]): List of callbacks to measure execution time for.
    """

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
