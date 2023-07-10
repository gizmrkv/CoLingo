from itertools import islice
from typing import Any, Callable, Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .callback import Callback


class Trainer(Callback):
    def __init__(
        self,
        game: nn.Module,
        optimizers: Iterable[optim.Optimizer],
        input: Iterable,
        loss: nn.Module,
        max_batches: int = 1,
    ):
        nn.ReLU6
        self._game = game
        self._optimizers = optimizers
        self._input = input
        self._loss = loss
        self._max_batches = max_batches

    def on_update(self, step: int):
        self._game.train()
        for optimizer in self._optimizers:
            optimizer.zero_grad()

        for input in islice(self._input, self._max_batches):
            output = self._game(input)
            loss = self._loss(output)
            loss.sum().backward()
            for optimizer in self._optimizers:
                optimizer.step()
