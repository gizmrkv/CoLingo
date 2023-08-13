from typing import Tuple

import torch.nn as nn
from torchtyping import TensorType

from ...game import IDecoder, IEncoder


class Encoder(nn.Module, IEncoder[TensorType[..., int], TensorType[..., float], None]):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def encode(
        self, input: TensorType[..., int]
    ) -> Tuple[TensorType[..., float], None]:
        return self.model(input), None


class Decoder(
    nn.Module,
    IDecoder[TensorType[..., float], TensorType[..., int], TensorType[..., float]],
):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def decode(
        self, latent: TensorType[..., float]
    ) -> Tuple[TensorType[..., int], TensorType[..., float]]:
        return self.model(latent)  # type: ignore
