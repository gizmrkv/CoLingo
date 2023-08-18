from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")
AE = TypeVar("AE")
AD = TypeVar("AD")


class IEncoder(ABC, Generic[T, U, A]):
    """
    Abstract base class for defining an encoder.
    """

    @abstractmethod
    def encode(self, input: T) -> Tuple[U, A]:
        """
        Encode the input data.

        Args:
            input (T): The input data.

        Returns:
            Tuple[U, A]: Encoded data and auxiliary information.
        """
        ...


class IDecoder(ABC, Generic[T, U, A]):
    """
    Abstract base class for defining a decoder.
    """

    @abstractmethod
    def decode(self, latent: T) -> Tuple[U, A]:
        """
        Decode the latent data.

        Args:
            latent (T): The latent data.

        Returns:
            Tuple[U, A]: Decoded data and auxiliary information.
        """
        ...


class IEncoderDecoder(IEncoder[T, U, AE], IDecoder[U, T, AD], Generic[T, U, AE, AD]):
    """
    Abstract base class for defining an encoder-decoder.
    """

    ...


@dataclass
class ReconstructionGameResult(Generic[T, U, AE, AD]):
    """
    Data class for storing the result of a reconstruction game.
    """

    encoder: IEncoder[T, U, AE]
    decoder: IDecoder[U, T, AD]
    input: T
    latent: U
    output: T
    encoder_aux: AE
    decoder_aux: AD


class ReconstructionGame(Generic[T, U, AE, AD]):
    """
    Class for managing a reconstruction game with an encoder and a decoder.
    """

    def __init__(
        self, encoder: IEncoder[T, U, AE], decoder: IDecoder[U, T, AD]
    ) -> None:
        """
        Initialize the ReconstructionGame.

        Args:
            encoder (IEncoder[T, U, AE]): An encoder instance.
            decoder (IDecoder[U, T, AD]): A decoder instance.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, input: T) -> ReconstructionGameResult[T, U, AE, AD]:
        """
        Run the reconstruction game.

        Args:
            input (T): The input data.

        Returns:
            ReconstructionGameResult[T, U, AE, AD]: Result of the reconstruction game.
        """
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
