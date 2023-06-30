import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..core import Callback
from ..logger import Logger


@dataclass
class SignalingGameResult:
    sender: Any
    receiver: Any
    input: Any

    latent_s: Any
    message_s: Any
    message_info_s: Any
    latent_r: Any
    output_r: Any
    output_info_r: Any

    output_s: Any | None = None
    output_info_s: Any | None = None
    message_r: Any | None = None
    message_info_r: Any | None = None
    latent_rec_s: Any | None = None
    latent_rec_r: Any | None = None


class SignalingGame(nn.Module):
    def __init__(
        self,
        run_sender_output: bool = False,
        run_receiver_send: bool = False,
        run_sender_recursion: bool = False,
        run_receiver_recursion: bool = False,
        sender_input_command: str = "input",
        sender_output_command: str = "output",
        sender_send_command: str = "send",
        sender_receive_command: str = "receive",
        receiver_input_command: str = "input",
        receiver_output_command: str = "output",
        receiver_send_command: str = "send",
        receiver_receive_command: str = "receive",
    ):
        super().__init__()
        self.run_sender_output = run_sender_output
        self.run_receiver_send = run_receiver_send
        self.run_sender_recursion = run_sender_recursion
        self.run_receiver_recursion = run_receiver_recursion
        self.sender_input_command = sender_input_command
        self.sender_output_command = sender_output_command
        self.sender_send_command = sender_send_command
        self.sender_receive_command = sender_receive_command
        self.receiver_input_command = receiver_input_command
        self.receiver_output_command = receiver_output_command
        self.receiver_send_command = receiver_send_command
        self.receiver_receive_command = receiver_receive_command

    def forward(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        input: torch.Tensor,
    ) -> SignalingGameResult:
        latent_s = sender(input=input, command=self.sender_input_command)
        message_s, message_info_s = sender(
            latent=latent_s, command=self.sender_send_command
        )

        latent_r = receiver(message=message_s, command=self.receiver_receive_command)
        output_r, output_info_r = receiver(
            latent=latent_r, command=self.receiver_output_command
        )

        result = SignalingGameResult(
            sender=sender,
            receiver=receiver,
            input=input,
            latent_s=latent_s,
            message_s=message_s,
            message_info_s=message_info_s,
            latent_r=latent_r,
            output_r=output_r,
            output_info_r=output_info_r,
        )

        if self.run_sender_output:
            result.output_s, result.output_info_s = sender(
                latent=latent_s, command=self.sender_output_command
            )

        if self.run_receiver_send:
            result.message_r, result.message_info_r = receiver(
                latent=latent_r, command=self.receiver_send_command
            )

        if self.run_sender_recursion:
            result.latent_rec_s = sender(
                message=message_s, command=self.sender_receive_command
            )

        if self.run_receiver_recursion:
            result.latent_rec_r = receiver(
                input=output_r, command=self.receiver_input_command
            )

        return result


class SignalingGameTrainer(Callback):
    def __init__(
        self,
        game: SignalingGame,
        agents: dict[str, nn.Module],
        optimizers: dict[str, optim.Optimizer],
        dataloader: DataLoader,
        loss: Callable[[SignalingGameResult, torch.Tensor], torch.Tensor],
        channels: list[(str, str)] | None = None,
        max_batches: int = 1,
    ):
        super().__init__()
        self.game = game
        self.agents = agents
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.loss = loss
        self.channels = channels
        self.max_batches = max_batches

        if self.channels is None:
            self.channels = []
            for sender_name in self.agents:
                for receiver_name in self.agents:
                    if sender_name != receiver_name:
                        self.channels.append((sender_name, receiver_name))

    def on_update(self, iteration: int):
        name_s, name_r = random.choice(self.channels)
        sender = self.agents[name_s]
        receiver = self.agents[name_r]
        optim_s = self.optimizers.get(name_s)
        optim_r = self.optimizers.get(name_r)

        sender.train()
        receiver.train()
        if optim_s is not None:
            optim_s.zero_grad()
        if optim_r is not None:
            optim_r.zero_grad()

        for input, target in islice(self.dataloader, self.max_batches):
            result: SignalingGameResult = self.game(
                sender=sender,
                receiver=receiver,
                input=input,
            )
            loss: torch.Tensor = self.loss(result, target)
            loss.sum().backward(retain_graph=True)

            if optim_s is not None:
                optim_s.step()
            if optim_r is not None:
                optim_r.step()


class SignalingGameEvaluator(Callback):
    def __init__(
        self,
        game: SignalingGame,
        agents: dict[str, nn.Module],
        input: torch.Tensor,
        target: torch.Tensor,
        metric: Callable[[SignalingGameResult], dict],
        logger: Logger | Iterable[Logger],
        name: str,
        channels: list[(str, str)] | None = None,
        run_on_begin: bool = True,
        run_on_end: bool = True,
    ):
        super().__init__()
        self.game = game
        self.agents = agents
        self.input = input
        self.target = target
        self.metric = metric
        self.loggers = [logger] if isinstance(logger, Logger) else logger
        self.name = name
        self.channels = channels
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end

        if self.channels is None:
            self.channels = []
            for sender_name in self.agents:
                for receiver_name in self.agents:
                    if sender_name != receiver_name:
                        self.channels.append((sender_name, receiver_name))

    def on_begin(self):
        if self.run_on_begin:
            self.evaluate()

    def on_end(self):
        if self.run_on_end:
            self.evaluate()

    def on_update(self, iteration: int):
        self.evaluate()

    def evaluate(self):
        log = {}
        for name_s, name_r in self.channels:
            sender = self.agents[name_s]
            receiver = self.agents[name_r]

            sender.eval()
            receiver.eval()

            with torch.no_grad():
                result = self.game(sender=sender, receiver=receiver, input=self.input)

            metric = self.metric(result)
            log |= {f"{name_s} -> {name_r}": metric}

        for logger in self.loggers:
            logger.log({self.name: log})
