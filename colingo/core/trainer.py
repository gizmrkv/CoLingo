from itertools import chain
from typing import Callable, Generic, Iterable, TypeVar

import torch
import torch.nn as nn
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
        loss: Callable[[int, T, Iterable[U]], TensorType[1, float]],
        optimizers: Iterable[torch.optim.Optimizer],
        device: str = "cuda",
        use_amp: bool = False,
    ) -> None:
        super().__init__()
        self.agents = agents
        self.input = input
        self.games = games
        self.loss = loss
        self.optimizers = optimizers
        self.device = device
        self.use_amp = use_amp

        self.parameters = chain.from_iterable(agent.parameters() for agent in agents)
        self.amp = torch.cuda.amp if device.startswith("cuda") else torch.cpu.amp
        self.scaler = self.amp.GradScaler(enabled=use_amp)

    def on_update(self, step: int) -> None:
        for agent in self.agents:
            agent.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        for input in self.input:
            with self.amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                outputs = [game(input) for game in self.games]
                loss = self.loss(step, input, outputs)

            self.scaler.scale(loss).backward()

            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(self.parameters, 1.0)

            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
                self.scaler.update()
