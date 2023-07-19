from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

BATCH = "batch"
OBJECT_LENGTH = "object_length"
OBJECT_N_VALUES = "object_n_values"
MESSAGE_LENGTH = "message_length"
MESSAGE_N_VALUES = "message_n_values"
LATENT_DIM = "latent_dim"


def pad_message(
    message: TensorType[BATCH, MESSAGE_LENGTH, int],
    logits: TensorType[BATCH, MESSAGE_LENGTH, MESSAGE_N_VALUES, float],
    eos: int = 0,
) -> tuple[
    TensorType[BATCH, MESSAGE_LENGTH, int],
    TensorType[BATCH, float],
    TensorType[BATCH, float],
    TensorType[BATCH, int],
]:
    distr = Categorical(logits=logits)
    log_prob = distr.log_prob(message)
    entropy = distr.entropy()

    mask = message == eos
    indices = torch.argmax(mask.int(), dim=1)
    no_mask = ~mask.any(dim=1)
    indices[no_mask] = message.shape[1]
    mask = torch.arange(message.shape[1]).expand(message.shape).to(message.device)
    mask = (mask <= indices.unsqueeze(-1)).long()

    length = mask.sum(dim=-1)
    message = message * mask
    log_prob = log_prob * mask
    entropy = entropy * mask

    return message, log_prob, entropy, length


@dataclass
class GameResult:
    sender: nn.Module
    receivers: nn.ModuleList
    input: TensorType[BATCH, OBJECT_LENGTH, int]

    latent_s: Any
    message_s: TensorType[BATCH, MESSAGE_LENGTH, int]
    message_logits_s: TensorType[BATCH, MESSAGE_LENGTH, MESSAGE_N_VALUES, float]
    message_log_prob_s: TensorType[BATCH, float]
    message_entropy_s: TensorType[BATCH, float]
    message_length_s: TensorType[BATCH, int]

    latent_r: list[Any]
    output_r: list[TensorType[BATCH, OBJECT_LENGTH, int]]
    output_logits_r: tuple[TensorType[BATCH, OBJECT_LENGTH, OBJECT_N_VALUES, float]]

    output_s: TensorType[BATCH, OBJECT_LENGTH, int] | None = None
    output_logits_s: TensorType[
        BATCH, OBJECT_LENGTH, OBJECT_N_VALUES, float
    ] | None = None

    messages_r: list[TensorType[BATCH, MESSAGE_LENGTH, int]] | None = None
    message_logits_r: tuple[
        TensorType[BATCH, MESSAGE_LENGTH, MESSAGE_N_VALUES, float]
    ] | None = None

    input_command: str = "input"
    output_command: str = "output"
    send_command: str = "send"
    receive_command: str = "receive"


class Game(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receivers: Sequence[nn.Module],
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

    def forward(self, input: TensorType[BATCH, OBJECT_LENGTH, int]) -> GameResult:
        ltt_s = self._sender(object=input, command=self._input_command)
        msg_s, msg_logits_s = self._sender(latent=ltt_s, command=self._send_command)
        msg_s, msg_log_prob_s, msg_entropy_s, msg_length_s = pad_message(
            msg_s, msg_logits_s
        )

        ltts_r = [
            receiver(message=msg_s, command=self._receive_command)
            for receiver in self._receivers
        ]
        output_r, output_logits_r = zip(
            *[
                receiver(latent=ltt_r, command=self._output_command)
                for receiver, ltt_r in zip(self._receivers, ltts_r)
            ]
        )
        result = GameResult(
            sender=self._sender,
            receivers=self._receivers,
            input=input,
            latent_s=ltt_s,
            message_s=msg_s,
            message_logits_s=msg_logits_s,
            message_log_prob_s=msg_log_prob_s,
            message_entropy_s=msg_entropy_s,
            message_length_s=msg_length_s,
            latent_r=ltts_r,
            output_r=output_r,
            output_logits_r=output_logits_r,
        )

        if self._run_sender_output:
            result.output_s, result.output_logits_s = self._sender(
                latent=ltt_s, command=self._output_command
            )

        if self._run_receiver_send:
            result.messages_r, result.message_logits_r = zip(
                *[
                    receiver(latent=ltt_r, command=self._send_command)
                    for receiver, ltt_r in zip(self._receivers, ltts_r)
                ]
            )

        return result
