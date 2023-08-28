from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchtyping import TensorType

from ...game import INetworkAgent


@dataclass
class MessageAuxiliary:
    max_len: int
    vocab_size: int
    message: TensorType[..., "max_len", int]
    logits: TensorType[..., "max_len", "vocab_size", float]
    log_prob: TensorType[..., "max_len", float]
    entropy: TensorType[..., "max_len", float]
    length: TensorType[..., int]


class Agent(
    nn.Module,
    INetworkAgent[
        TensorType[..., int],
        TensorType[..., float],
        TensorType[..., int],
        None,
        TensorType[..., float],
        None,
        MessageAuxiliary,
    ],
):
    def __init__(
        self,
        object_encoder: nn.Module,
        object_decoder: nn.Module,
        message_encoder: nn.Module,
        message_decoder: nn.Module,
        eos: int = 0,
    ) -> None:
        super().__init__()
        self.object_encoder = object_encoder
        self.object_decoder = object_decoder
        self.message_encoder = message_encoder
        self.message_decoder = message_decoder
        self.eos = eos

    def encode_object(
        self, input: TensorType[..., int]
    ) -> Tuple[TensorType[..., float], None]:
        return self.object_encoder(input), None

    def encode_message(
        self, message: TensorType[..., int]
    ) -> Tuple[TensorType[..., float], None]:
        return self.message_encoder(message), None

    def decode_object(
        self,
        latent: TensorType[..., float],
        input: TensorType[..., int] | None = None,
        message: TensorType[..., int] | None = None,
    ) -> Tuple[TensorType[..., int], TensorType[..., float]]:
        return self.object_decoder(latent)

    def decode_message(
        self,
        latent: TensorType[..., float],
        input: TensorType[..., int] | None = None,
        message: TensorType[..., int] | None = None,
    ) -> Tuple[TensorType[..., int], MessageAuxiliary]:
        message, logits = self.message_decoder(latent)

        distr = Categorical(logits=logits)
        log_prob = distr.log_prob(message)
        entropy = distr.entropy()

        mask = message == self.eos
        indices = torch.argmax(mask.int(), dim=1)
        no_mask = ~mask.any(dim=1)
        indices[no_mask] = message.shape[1]
        mask = torch.arange(message.shape[1]).expand(message.shape).to(message.device)
        mask = (mask <= indices.unsqueeze(-1)).long()

        length = mask.sum(dim=-1)
        message = message * mask
        log_prob = log_prob * mask
        entropy = entropy * mask

        return message, MessageAuxiliary(
            max_len=message.shape[1],
            vocab_size=logits.shape[-1],
            message=message,
            logits=logits,
            log_prob=log_prob,
            entropy=entropy,
            length=length,
        )
