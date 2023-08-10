from typing import Literal

from torch import nn
from torchtyping import TensorType


class Agent(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    N_VALUES = "n_values"
    LATENT_DIM = "latent_dim"

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(
        self,
        object: TensorType[BATCH, LENGTH, int] | None = None,
        latent: TensorType[BATCH, LATENT_DIM, float] | None = None,
        command: Literal["input", "output"] = "input",
    ) -> (
        TensorType[BATCH, LATENT_DIM, float]
        | tuple[
            TensorType[BATCH, LENGTH, int], TensorType[BATCH, LENGTH, N_VALUES, float]
        ]
    ):
        match command:
            case "input":
                return self._encoder(object)
            case "output":
                return self._decoder(latent)
            case _:
                raise ValueError(f"Unknown command: {command}")
