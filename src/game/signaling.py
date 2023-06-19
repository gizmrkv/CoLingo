import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable, Tuple

import torch as th
from torch.utils.data import DataLoader

from ..core import Callback
from ..logger import Logger


@dataclass
class SignalingGameResult:
    input: th.Tensor
    target: th.Tensor
    sender_message: Any
    receiver_output: th.Tensor
    sender_loss: th.Tensor | None = None
    receiver_loss: th.Tensor | None = None
    total_loss: th.Tensor | None = None

    # receiver_echo
    receiver_echo: Any | None = None
    receiver_echo_loss: th.Tensor | None = None

    # sender_internal
    sender_internal: Any | None = None
    sender_internal_loss: th.Tensor | None = None


@dataclass
class SignalingGameOption:
    receiver_echo: bool = False
    receiver_echo_loss: th.nn.Module | None = None
    receiver_echo_command: str = "echo"
    sender_internal: bool = False
    sender_internal_loss: th.nn.Module | None = None
    sender_internal_input_command: str = "input"


class SignalingGame(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        sender: th.nn.Module,
        receiver: th.nn.Module,
        input: th.Tensor,
        target: th.Tensor,
        output_loss: th.nn.Module | None = None,
        message_loss: th.nn.Module | None = None,
        sender_input_command: str = "input",
        receiver_input_command: str = "input",
        output_command: str = "output",
        message_command: str = "message",
        option: SignalingGameOption | None = None,
    ) -> SignalingGameResult:
        option = option or SignalingGameOption()

        hidden_s = sender(input=input, command=sender_input_command)
        message_s = sender(hidden=hidden_s, command=message_command)

        hidden_r = receiver(message=message_s, command=receiver_input_command)
        output_r = receiver(hidden=hidden_r, command=output_command)

        result = SignalingGameResult(
            input=input,
            target=target,
            sender_message=message_s,
            receiver_output=output_r,
        )

        if option.receiver_echo:
            result.receiver_echo = receiver(
                hidden=hidden_r, command=option.receiver_echo_command
            )

        if option.sender_internal:
            result.sender_internal = sender(
                message=message_s, command=option.sender_internal_input_command
            )

        if self.training:
            result.receiver_loss = output_loss(output_r, target)
            result.sender_loss = message_loss(message_s, result.receiver_loss)

            if option.receiver_echo:
                result.receiver_echo_loss = option.receiver_echo_loss(
                    result.receiver_echo, message_s
                )

            if option.sender_internal:
                result.sender_internal_loss = option.sender_internal_loss(
                    hidden_s, result.sender_internal
                )

            result.total_loss = result.sender_loss + result.receiver_loss

            if option.receiver_echo:
                result.total_loss += result.receiver_echo_loss

            if option.sender_internal:
                result.total_loss += result.sender_internal_loss

        return result


class SignalingGameTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, th.nn.Module],
        optimizers: dict[str, th.optim.Optimizer],
        dataloader: DataLoader,
        output_loss: th.nn.Module,
        message_loss: th.nn.Module,
        channels: list[(str, str)] | None = None,
        max_batches: int = 1,
        sender_input_command: str = "input",
        receiver_input_command: str = "input",
        output_command: str = "output",
        message_command: str = "message",
        option: SignalingGameOption | None = None,
    ):
        super().__init__()
        self.game = SignalingGame()
        self.agents = agents
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.output_loss = output_loss
        self.message_loss = message_loss
        self.channels = channels
        self.max_batches = max_batches
        self.sender_input_command = sender_input_command
        self.receiver_input_command = receiver_input_command
        self.output_command = output_command
        self.message_command = message_command
        self.option = option

        if self.channels is None:
            self.channels = []
            for sender_name in self.agents:
                for receiver_name in self.agents:
                    if sender_name != receiver_name:
                        self.channels.append((sender_name, receiver_name))

    def on_update(self, iteration: int):
        sender_name, receiver_name = random.choice(self.channels)
        sender = self.agents[sender_name]
        receiver = self.agents[receiver_name]
        sender_optimizer = self.optimizers[sender_name]
        receiver_optimizer = self.optimizers[receiver_name]

        self.game.train()
        sender.train()
        receiver.train()
        sender_optimizer.zero_grad()
        receiver_optimizer.zero_grad()

        for input, target in islice(self.dataloader, self.max_batches):
            result: SignalingGameResult = self.game(
                sender=sender,
                receiver=receiver,
                input=input,
                target=target,
                output_loss=self.output_loss,
                message_loss=self.message_loss,
                sender_input_command=self.sender_input_command,
                receiver_input_command=self.receiver_input_command,
                output_command=self.output_command,
                message_command=self.message_command,
                option=self.option,
            )
            result.total_loss.sum().backward(retain_graph=True)
            sender_optimizer.step()
            receiver_optimizer.step()


class SignalingGameEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, th.nn.Module],
        input: th.Tensor,
        target: th.Tensor,
        metric: Callable[[SignalingGameResult], dict],
        logger: Logger | Iterable[Logger],
        name: str,
        channels: list[(str, str)] | None = None,
        run_on_begin: bool = True,
        run_on_end: bool = True,
        sender_input_command: str = "input",
        receiver_input_command: str = "input",
        output_command: str = "output",
        message_command: str = "message",
        option: SignalingGameOption | None = None,
    ):
        super().__init__()
        self.game = SignalingGame()
        self.agents = agents
        self.input = input
        self.target = target
        self.metric = metric
        self.loggers = [logger] if isinstance(logger, Logger) else logger
        self.name = name
        self.channels = channels
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end
        self.sender_input_command = sender_input_command
        self.receiver_input_command = receiver_input_command
        self.output_command = output_command
        self.message_command = message_command
        self.option = option

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
        for sender_name, receiver_name in self.channels:
            sender = self.agents[sender_name]
            receiver = self.agents[receiver_name]

            self.game.eval()
            sender.eval()
            receiver.eval()

            with th.no_grad():
                result = self.game(
                    sender=sender,
                    receiver=receiver,
                    input=self.input,
                    target=self.target,
                    sender_input_command=self.sender_input_command,
                    receiver_input_command=self.receiver_input_command,
                    output_command=self.output_command,
                    message_command=self.message_command,
                    option=self.option,
                )

            metric = self.metric(result)
            log |= {f"{sender_name} -> {receiver_name}": metric}

        for logger in self.loggers:
            logger.log({self.name: log})
