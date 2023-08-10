from dataclasses import dataclass
from itertools import product
from typing import Callable, Generic, Iterable, Protocol, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchtyping import TensorType

from .runner import RunnerCallback

T = TypeVar("T")
U = TypeVar("U")


class Trainer(RunnerCallback, Generic[T, U]):
    def __init__(
        self,
        agents: Iterable[nn.Module],
        input: Iterable[T],
        games: Iterable[Callable[[T], U]],
        loss: Callable[[U], TensorType[..., float]],
        optimizers: Iterable[torch.optim.Optimizer],
    ) -> None:
        super().__init__()
        self.agents = agents
        self.input = input
        self.games = games
        self.loss = loss
        self.optimizers = optimizers

    def on_update(self, step: int) -> None:
        for agent in self.agents:
            agent.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        for input in self.input:
            outputs = [game(input) for game in self.games]
            losses = [self.loss(output) for output in outputs]
            loss = torch.stack(losses).mean()
            loss.backward(retain_graph=True)
            for optimizer in self.optimizers:
                optimizer.step()

    def on_begin(self) -> None:
        pass

    def on_end(self) -> None:
        pass
