import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable

import torch as th
from torch.utils.data import DataLoader

from ..core import Callback
from ..logger import Logger


@dataclass
class SignalingGameResult:
    sender: Any
    receiver: Any
    input: Any
    target: Any

    sender_latent: Any
    sender_message: Any
    receiver_latent: Any
    receiver_output: Any

    sender_output: Any | None = None
    receiver_message: Any | None = None


class SignalingGame(th.nn.Module):
    def __init__(
        self,
        run_sender_output: bool = False,
        run_receiver_send: bool = False,
        sender_input_command: str = "input",
        sender_output_command: str = "output",
        sender_send_command: str = "send",
        receiver_output_command: str = "output",
        receiver_receive_command: str = "receive",
        receiver_send_command: str = "send",
    ):
        super().__init__()
        self.run_sender_output = run_sender_output
        self.run_receiver_send = run_receiver_send
        self.sender_input_command = sender_input_command
        self.sender_output_command = sender_output_command
        self.sender_send_command = sender_send_command
        self.receiver_output_command = receiver_output_command
        self.receiver_receive_command = receiver_receive_command
        self.receiver_send_command = receiver_send_command

    def forward(
        self,
        sender: th.nn.Module,
        receiver: th.nn.Module,
        input: th.Tensor,
        target: th.Tensor,
    ) -> SignalingGameResult:
        latent_s = sender(input=input, command=self.sender_input_command)
        message_s = sender(latent=latent_s, command=self.sender_send_command)

        latent_r = receiver(message=message_s, command=self.receiver_receive_command)
        output_r = receiver(latent=latent_r, command=self.receiver_output_command)

        result = SignalingGameResult(
            sender=sender,
            receiver=receiver,
            input=input,
            target=target,
            sender_latent=latent_s,
            sender_message=message_s,
            receiver_latent=latent_r,
            receiver_output=output_r,
        )

        if self.run_sender_output:
            result.sender_output = sender(
                latent=latent_s, command=self.sender_output_command
            )

        if self.run_receiver_send:
            result.receiver_message = receiver(
                latent=latent_r, command=self.receiver_send_command
            )

        return result


class SignalingGameTrainer(Callback):
    def __init__(
        self,
        game: SignalingGame,
        agents: dict[str, th.nn.Module],
        optimizers: dict[str, th.optim.Optimizer],
        dataloader: DataLoader,
        loss: th.nn.Module,
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
        optim_s = self.optimizers[name_s] if name_s in self.optimizers else None
        optim_r = self.optimizers[name_r] if name_r in self.optimizers else None

        sender.train()
        receiver.train()
        optim_s.zero_grad()
        optim_r.zero_grad()

        for input, target in islice(self.dataloader, self.max_batches):
            # result: SignalingGameResult = self.game(
            #     sender=sender,
            #     receiver=receiver,
            #     input=input,
            #     target=target,
            # )
            # self.loss(result).backward(retain_graph=True)
            result1: SignalingGameResult = self.game(
                sender=sender,
                receiver=receiver,
                input=input,
                target=target,
            )
            result2: SignalingGameResult = self.game(
                sender=receiver,
                receiver=sender,
                input=input,
                target=target,
            )
            loss1 = self.loss(result1)
            loss2 = self.loss(result2)
            (loss1 + loss2).backward(retain_graph=True)

            if optim_s is not None:
                optim_s.step()
            if optim_r is not None:
                optim_r.step()


class SignalingGameEvaluator(Callback):
    def __init__(
        self,
        game: SignalingGame,
        agents: dict[str, th.nn.Module],
        input: th.Tensor,
        target: th.Tensor,
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
        for sender_name, receiver_name in self.channels:
            sender = self.agents[sender_name]
            receiver = self.agents[receiver_name]

            sender.eval()
            receiver.eval()

            with th.no_grad():
                result = self.game(
                    sender=sender,
                    receiver=receiver,
                    input=self.input,
                    target=self.target,
                )

            metric = self.metric(result)
            log |= {f"{sender_name} -> {receiver_name}": metric}

        for logger in self.loggers:
            logger.log({self.name: log})
