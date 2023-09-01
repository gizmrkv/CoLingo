import datetime
import json
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchtyping import TensorType

from ...game import IDecoder, IEncoder


@dataclass
class MessageAuxiliary:
    max_len: int
    vocab_size: int
    message: TensorType[..., "max_len", int]
    logits: TensorType[..., "max_len", "vocab_size", float]
    log_prob: TensorType[..., "max_len", float]
    entropy: TensorType[..., "max_len", float]
    length: TensorType[..., int]


class Sender(
    nn.Module,
    IEncoder[TensorType[..., int], TensorType[..., int], MessageAuxiliary],
):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, eos: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.eos = eos

    def encode(
        self, input: TensorType[..., int]
    ) -> Tuple[TensorType[..., int], MessageAuxiliary]:
        latent = self.encoder(input)
        message, logits = self.decoder(latent)

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


class Receiver(
    nn.Module,
    IDecoder[TensorType[..., int], TensorType[..., int], TensorType[..., float]],
):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def decode(
        self, latent: TensorType[..., float]
    ) -> Tuple[TensorType[..., int], TensorType[..., float]]:
        latent = self.encoder(latent)
        output, aux = self.decoder(latent)
        return output, aux
