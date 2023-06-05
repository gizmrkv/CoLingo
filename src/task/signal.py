import random
from itertools import islice
from typing import Iterable

import torch as th
from torch.utils.data import DataLoader

from ..agent import Agent
from ..core.callback import Callback
from ..logger import Logger
from ..metric import Metric


class SignalTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        optimizers: dict[str, th.optim.Optimizer],
        dataloader: DataLoader,
        sender_loss: th.nn.Module,
        receiver_loss: th.nn.Module,
        sender_input_key,
        sender_output_key,
        receiver_input_key,
        receiver_output_key,
        max_batches: int = 1,
        channels: list[tuple[str, str]] | None = None,
    ):
        super().__init__()

        self.agents = agents
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.sender_loss = sender_loss
        self.receiver_loss = receiver_loss
        self.sender_input_key = sender_input_key
        self.sender_output_key = sender_output_key
        self.receiver_input_key = receiver_input_key
        self.receiver_output_key = receiver_output_key
        self.max_batches = max_batches

        self.agents_names = list(self.agents.keys())

        if channels is None:
            self.channels = []
            for sender_name in self.agents_names:
                for receiver_name in self.agents_names:
                    if sender_name != receiver_name:
                        self.channels.append((sender_name, receiver_name))
        else:
            self.channels = channels

    def on_update(self, iteration: int):
        for input, target in islice(self.dataloader, self.max_batches):
            sender_name, receiver_name = random.choice(self.channels)
            sender = self.agents[sender_name]
            receiver = self.agents[receiver_name]

            sender.train()
            receiver.train()
            self.optimizers[sender_name].zero_grad()
            self.optimizers[receiver_name].zero_grad()

            hidden_s = sender.input({self.sender_input_key: input})
            ((message, logprob, entropy, length),) = sender.output(
                self.sender_output_key, hidden=hidden_s
            )
            hidden_r = receiver.input({self.receiver_input_key: message})
            (output,) = receiver.output(self.receiver_output_key, hidden=hidden_r)

            receiver_loss = self.receiver_loss(input=output, target=target)
            sender_loss = self.sender_loss(
                loss=receiver_loss, logprob=logprob, entropy=entropy, length=length
            )
            loss: th.Tensor = (sender_loss + receiver_loss).mean()

            loss.backward(retain_graph=True)
            self.optimizers[sender_name].step()
            self.optimizers[receiver_name].step()


class SignalEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        input: th.Tensor,
        target: th.Tensor,
        metrics: Iterable[Metric],
        loggers: Iterable[Logger],
        sender_input_key,
        sender_output_key,
        receiver_input_key,
        receiver_output_key,
        name: str,
        interval: int = 1,
        channels: list[tuple[str, str]] | None = None,
    ):
        super().__init__()
        self.agents = agents
        self.input = input
        self.target = target
        self.metircs = metrics
        self.loggers = loggers
        self.sender_input_key = sender_input_key
        self.sender_output_key = sender_output_key
        self.receiver_input_key = receiver_input_key
        self.receiver_output_key = receiver_output_key
        self.name = name
        self.interval = interval

        self._count = 0

        self.agents_names = list(self.agents.keys())

        if channels is None:
            self.channels = []
            for sender_name in self.agents_names:
                for receiver_name in self.agents_names:
                    if sender_name != receiver_name:
                        self.channels.append((sender_name, receiver_name))
        else:
            self.channels = channels

    def on_update(self, iteration: int):
        if self._count % self.interval != 0:
            return

        self._count += 1

        def channel_name(channel: tuple[str, str]) -> str:
            return f"{channel[0]} -> {channel[1]}"

        logs = {channel_name(channel): {} for channel in self.channels}
        for channel in self.channels:
            sender_name, receiver_name = channel
            sender = self.agents[sender_name]
            receiver = self.agents[receiver_name]

            sender.eval()
            receiver.eval()

            with th.no_grad():
                hidden_s = sender.input({self.sender_input_key: self.input})
                ((message, log_prob, entropy, length),) = sender.output(
                    self.sender_output_key, hidden=hidden_s
                )
                hidden_r = receiver.input({self.receiver_input_key: message})
                (output,) = receiver.output(self.receiver_output_key, hidden=hidden_r)

            for metric in self.metircs:
                met = metric.calculate(
                    input=self.input,
                    message=message,
                    output=output,
                    target=self.target,
                    log_prob=log_prob,
                    entropy=entropy,
                    length=length,
                )
                logs[channel_name(channel)] |= met

        for logger in self.loggers:
            logger.log({self.name: logs})
