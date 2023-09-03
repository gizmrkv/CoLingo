from dataclasses import dataclass

from torch import nn
from torchtyping import TensorType

from ...core import Playable


@dataclass
class ReconstructionGameResult:
    encoder: nn.Module
    decoder: nn.Module
    input: TensorType[..., int]
    latent: TensorType[..., float]
    output: TensorType[..., int]
    logits: TensorType[..., float]


class ReconstructionGame(Playable[TensorType[..., int], ReconstructionGameResult]):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def play(
        self, input: TensorType[..., int], step: int | None = None
    ) -> ReconstructionGameResult:
        latent = self.encoder(input)
        output, logits = self.decoder(latent, concept=input)
        return ReconstructionGameResult(
            encoder=self.encoder,
            decoder=self.decoder,
            input=input,
            latent=latent,
            output=output,
            logits=logits,
        )
