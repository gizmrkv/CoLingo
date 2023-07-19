from typing import Literal

from torch import nn
from torchtyping import TensorType


class Agent(nn.Module):
    BATCH = "batch"
    OBJECT_LENGTH = "object_length"
    OBJECT_N_VALUES = "object_n_values"
    MESSAGE_LENGTH = "message_length"
    MESSAGE_N_VALUES = "message_n_values"
    LATENT_DIM = "latent_dim"

    def __init__(
        self,
        object_encoder: nn.Module,
        object_decoder: nn.Module,
        message_encoder: nn.Module,
        message_decoder: nn.Module,
        shared: nn.Module | None = None,
    ):
        super().__init__()
        self._object_encoder = object_encoder
        self._object_decoder = object_decoder
        self._message_encoder = message_encoder
        self._message_decoder = message_decoder
        self._shared = shared

    def forward(
        self,
        object: TensorType[BATCH, OBJECT_LENGTH, int] | None = None,
        message: TensorType[BATCH, MESSAGE_LENGTH, int] | None = None,
        latent: TensorType[BATCH, LATENT_DIM, float] | None = None,
        command: Literal["input", "output", "receive", "send", "echo"] = "input",
    ) -> (
        TensorType[BATCH, LATENT_DIM, float]
        | tuple[
            TensorType[BATCH, OBJECT_LENGTH, int],
            TensorType[BATCH, OBJECT_LENGTH, OBJECT_N_VALUES, float],
        ]
        | tuple[
            TensorType[BATCH, MESSAGE_LENGTH, int],
            TensorType[BATCH, MESSAGE_LENGTH, MESSAGE_N_VALUES, float],
        ]
    ):
        match command:
            case "input":
                latent = self._object_encoder(object)
                latent = self._shared(latent) if self._shared else latent
                return latent
            case "output":
                return self._object_decoder(latent)
            case "receive":
                latent = self._message_encoder(message)
                latent = self._shared(latent) if self._shared else latent
                return latent
            case "send":
                return self._message_decoder(latent)
            case "echo":
                return self._message_decoder(latent, message)
            case _:
                raise ValueError(f"Unknown command: {command}")
