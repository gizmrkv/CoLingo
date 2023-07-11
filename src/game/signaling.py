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
    sender: nn.Module
    receivers: Iterable[nn.Module]
    input: torch.Tensor

    latent_s: torch.Tensor
    message_s: torch.Tensor
    message_info_s: Any

    latents_r: tuple[torch.Tensor]
    outputs_r: tuple[torch.Tensor]
    output_infos_r: tuple[Any]

    output_s: torch.Tensor | None = None
    output_info_s: Any | None = None
    messages_r: tuple[torch.Tensor] | None = None
    message_infos_r: tuple[Any] | None = None


class SignalingGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receivers: Iterable[nn.Module],
        run_sender_output: bool = False,
        run_receiver_send: bool = False,
        input_command: str = "input",
        output_command: str = "output",
        send_command: str = "send",
        receive_command: str = "receive",
    ):
        super().__init__()
        self._sender = sender
        self._receivers = nn.ModuleList(receivers)
        self._run_sender_output = run_sender_output
        self._run_receiver_send = run_receiver_send
        self._input_command = input_command
        self._output_command = output_command
        self._send_command = send_command
        self._receive_command = receive_command

    def forward(self, input: torch.Tensor) -> SignalingGameResult:
        ltt_s = self._sender(input=input, command=self._input_command)
        msg_s, msg_info_s = self._sender(latent=ltt_s, command=self._send_command)

        ltts_r = [
            receiver(message=msg_s, command=self._receive_command)
            for receiver in self._receivers
        ]
        outputs_r, output_infos_r = zip(
            *[
                receiver(latent=ltt_r, command=self._output_command)
                for receiver, ltt_r in zip(self._receivers, ltts_r)
            ]
        )

        result = SignalingGameResult(
            sender=self._sender,
            receivers=self._receivers,
            input=input,
            latent_s=ltt_s,
            message_s=msg_s,
            message_info_s=msg_info_s,
            latents_r=ltts_r,
            outputs_r=outputs_r,
            output_infos_r=output_infos_r,
        )

        if self._run_sender_output:
            result.output_s, result.output_info_s = self._sender(
                latent=ltt_s, command=self._output_command
            )

        if self._run_receiver_send:
            result.messages_r, result.message_infos_r = zip(
                *[
                    receiver(latent=ltt_r, command=self._send_command)
                    for receiver, ltt_r in zip(self._receivers, ltts_r)
                ]
            )

        return result
