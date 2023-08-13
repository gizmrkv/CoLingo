from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from .reconstruction import IDecoder, IEncoder

T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")
AE = TypeVar("AE")
AD = TypeVar("AD")


@dataclass
class ReconstructionBroadcastGameResult(Generic[T, U, AE, AD]):
    encoder: IEncoder[T, U, AE]
    decoders: Sequence[IDecoder[U, T, AD]]
    input: T
    latent: U
    outputs: Sequence[T]
    encoder_aux: AE
    decoders_aux: Sequence[AD]


class ReconstructionBroadcastGame(Generic[T, U, AE, AD]):
    def __init__(
        self, encoder: IEncoder[T, U, AE], decoders: Sequence[IDecoder[U, T, AD]]
    ) -> None:
        self.encoder = encoder
        self.decoders = decoders

    def __call__(self, input: T) -> ReconstructionBroadcastGameResult[T, U, AE, AD]:
        latent, enc_aux = self.encoder.encode(input)
        outputs, decs_aux = zip(*[decoder.decode(latent) for decoder in self.decoders])
        return ReconstructionBroadcastGameResult(
            encoder=self.encoder,
            decoders=self.decoders,
            input=input,
            latent=latent,
            outputs=outputs,
            encoder_aux=enc_aux,
            decoders_aux=decs_aux,
        )
