import random
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterable, Tuple

import torch as th
from torch.utils.data import DataLoader

from ..core import Callback
from ..logger import Logger
from ..loss import MessageLoss


@dataclass
class MessageSignalingGameResult:
    input: th.Tensor
    target: th.Tensor
    sender_message: th.Tensor
    sender_log_prob: th.Tensor
    sender_entropy: th.Tensor
    sender_length: th.Tensor
    receiver_output: th.Tensor
    sender_output: th.Tensor | None = None
    sender_output_loss: th.Tensor | None = None
    receiver_message: th.Tensor | None = None
    receiver_log_prob: th.Tensor | None = None
    receiver_entropy: th.Tensor | None = None
    receiver_length: th.Tensor | None = None
    receiver_echo_loss: th.Tensor | None = None
    sender_loss: th.Tensor | None = None
    receiver_loss: th.Tensor | None = None
    total_loss: th.Tensor | None = None


class MessageSignalingGame(th.nn.Module):
    def __init__(
        self,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        loss_baseline: th.nn.Module | None = None,
        length_baseline: th.nn.Module | None = None,
        name: str = "MessageSignalingGame",
    ):
        super().__init__()
        self.message_loss = MessageLoss(
            entropy_weight=entropy_weight,
            length_weight=length_weight,
            baseline=loss_baseline,
            length_baseline=length_baseline,
        )
        self.name = name

    def forward(
        self,
        sender: th.nn.Module,
        receiver: th.nn.Module,
        input: th.Tensor,
        target: th.Tensor,
        loss: th.nn.Module | None = None,
        sender_output: bool = False,
        receiver_echo: bool = False,
        input_command: str = "input",
        output_command: str = "output",
        message_command: str = "message",
        echo_command: str = "echo",
    ) -> Tuple[th.Tensor, MessageSignalingGameResult] | MessageSignalingGameResult:
        hidden_s = sender(input=input, command=input_command)
        message_s, prob_s, log_prob_s, entropy_s, length_s = sender(
            hidden=hidden_s, command=message_command
        )

        hidden_r = receiver(message=message_s, command=input_command)
        output_r = receiver(hidden=hidden_r, command=output_command)

        result = MessageSignalingGameResult(
            input=input,
            target=target,
            sender_message=message_s,
            sender_log_prob=log_prob_s,
            sender_entropy=entropy_s,
            sender_length=length_s,
            receiver_output=output_r,
        )

        if self.training:
            receiver_loss = loss(output_r, target)
            sender_loss = self.message_loss(
                loss=receiver_loss,
                log_prob=log_prob_s,
                entropy=entropy_s,
                length=length_s,
            )

        if sender_output:
            output_s = sender(hidden=hidden_s, command=output_command)
            result.sender_output = output_s
            if self.training:
                sender_output_loss = loss(output_s, target)
                sender_loss = sender_loss + sender_output_loss
                result.sender_output_loss = sender_output_loss

        if receiver_echo:
            message_r, prob_r, log_prob_r, entropy_r, length_r = receiver(
                hidden=hidden_r, command=echo_command
            )
            result.receiver_message = message_r
            result.receiver_log_prob = log_prob_r
            result.receiver_entropy = entropy_r
            result.receiver_length = length_r

            if self.training:
                # receiver_echo_loss = -(
                #     (message_r == message_s).float() * log_prob_r
                # ).mean(dim=-1)
                receiver_echo_loss = 1.2 * th.nn.functional.cross_entropy(
                    prob_r.reshape(-1, 50), message_s[:, :-1].reshape(-1)
                )
                receiver_loss = receiver_loss + receiver_echo_loss
                result.receiver_echo_loss = receiver_echo_loss

        if self.training:
            total_loss = sender_loss + receiver_loss
            result.sender_loss = sender_loss
            result.receiver_loss = receiver_loss
            result.total_loss = total_loss
            return total_loss, result
        else:
            return result


class MessageSignalingGameTrainer(Callback):
    def __init__(
        self,
        game: MessageSignalingGame,
        loss: th.nn.Module,
        agents: dict[str, th.nn.Module],
        optimizers: dict[str, th.optim.Optimizer],
        dataloader: DataLoader,
        channels: list[(str, str)] | None = None,
        max_batches: int = 1,
        sender_output: bool = False,
        receiver_echo: bool = False,
        input_command: str = "input",
        output_command: str = "output",
        message_command: str = "message",
        echo_command: str = "echo",
    ):
        super().__init__()
        self.game = game
        self.loss = loss
        self.agents = agents
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.channels = channels
        self.max_batches = max_batches
        self.sender_output = sender_output
        self.receiver_echo = receiver_echo
        self.input_command = input_command
        self.output_command = output_command
        self.message_command = message_command
        self.echo_command = echo_command

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
            loss, _ = self.game(
                sender=sender,
                receiver=receiver,
                input=input,
                target=target,
                loss=self.loss,
                sender_output=self.sender_output,
                receiver_echo=self.receiver_echo,
                input_command=self.input_command,
                output_command=self.output_command,
                message_command=self.message_command,
                echo_command=self.echo_command,
            )
            loss.sum().backward(retain_graph=True)
            sender_optimizer.step()
            receiver_optimizer.step()


class MessageSignalingGameEvaluator(Callback):
    def __init__(
        self,
        game: MessageSignalingGame,
        agents: dict[str, th.nn.Module],
        input: th.Tensor,
        target: th.Tensor,
        metric: Callable[[MessageSignalingGameResult], dict],
        logger: Logger | Iterable[Logger],
        name: str,
        channels: list[(str, str)] | None = None,
        sender_output: bool = False,
        receiver_echo: bool = False,
        run_on_begin: bool = True,
        run_on_end: bool = True,
        input_command: str = "input",
        output_command: str = "output",
        message_command: str = "message",
        echo_command: str = "echo",
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
        self.sender_output = sender_output
        self.receiver_echo = receiver_echo
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end
        self.input_command = input_command
        self.output_command = output_command
        self.message_command = message_command
        self.echo_command = echo_command

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
                    sender_output=self.sender_output,
                    receiver_echo=self.receiver_echo,
                    input_command=self.input_command,
                    output_command=self.output_command,
                    message_command=self.message_command,
                    echo_command=self.echo_command,
                )

            metric = self.metric(result)
            log |= {f"{sender_name} -> {receiver_name}": metric}

        for logger in self.loggers:
            logger.log({self.name: log})
