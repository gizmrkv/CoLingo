from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")
AE = TypeVar("AE")
AD = TypeVar("AD")


class IEncoder(ABC, Generic[T, U, A]):
    @abstractmethod
    def encode(self, input: T) -> Tuple[U, A]:
        ...


class IDecoder(ABC, Generic[T, U, A]):
    @abstractmethod
    def decode(self, latent: T) -> Tuple[U, A]:
        ...


@dataclass
class ReconstructionGameResult(Generic[T, U, AE, AD]):
    encoder: IEncoder[T, U, AE]
    decoder: IDecoder[U, T, AD]
    input: T
    latent: U
    output: T
    encoder_aux: AE
    decoder_aux: AD


class ReconstructionGame(Generic[T, U, AE, AD]):
    def __init__(
        self, encoder: IEncoder[T, U, AE], decoder: IDecoder[U, T, AD]
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, input: T) -> ReconstructionGameResult[T, U, AE, AD]:
        latent, enc_aux = self.encoder.encode(input)
        output, dec_aux = self.decoder.decode(latent)
        return ReconstructionGameResult(
            encoder=self.encoder,
            decoder=self.decoder,
            input=input,
            latent=latent,
            output=output,
            encoder_aux=enc_aux,
            decoder_aux=dec_aux,
        )
