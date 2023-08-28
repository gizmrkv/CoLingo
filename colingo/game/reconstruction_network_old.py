from dataclasses import dataclass
from typing import Dict, Generic, Mapping, Set, TypeVar

from .reconstruction import IDecoder, IEncoder, IEncoderDecoder

T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")
AE = TypeVar("AE")
AD = TypeVar("AD")


@dataclass
class ReconstructionNetworkSubGameResult(Generic[T, U, AE, AD]):
    """
    The result of a sub-game within a ReconstructionNetworkGame.

    Attributes:
        encoder (IEncoder[T, U, AE]): The encoder used in the sub-game.
        decoders (Dict[str, IDecoder[U, T, AD]]): The decoders used in the sub-game.
        input (T): The input data to the sub-game.
        latent (U): The latent representation obtained from encoding the input.
        outputs (Dict[str, T]): The decoded outputs from different decoders.
        encoder_aux (AE): Auxiliary data obtained during encoding.
        decoders_aux (Dict[str, AD]): Auxiliary data obtained during decoding.
    """

    encoder: IEncoder[T, U, AE]
    decoders: Dict[str, IDecoder[U, T, AD]]
    input: T
    latent: U
    outputs: Dict[str, T]
    encoder_aux: AE
    decoders_aux: Dict[str, AD]


class ReconstructionNetworkSubGame(Generic[T, U, AE, AD]):
    """
    A sub-game within the ReconstructionNetworkGame.

    Args:
        encoder (IEncoder[T, U, AE]): The encoder used in the sub-game.
        decoders (Dict[str, IDecoder[U, T, AD]]): The decoders used in the sub-game.
    """

    def __init__(
        self, encoder: IEncoder[T, U, AE], decoders: Dict[str, IDecoder[U, T, AD]]
    ) -> None:
        self.encoder = encoder
        self.decoders = decoders

    def __call__(self, input: T) -> ReconstructionNetworkSubGameResult[T, U, AE, AD]:
        """
        Execute the sub-game on the given input data.

        Args:
            input (T): The input data.

        Returns:
            ReconstructionNetworkSubGameResult[T, U, AE, AD]: The result of the sub-game.
        """
        latent, enc_aux = self.encoder.encode(input)
        outputs = {}
        decs_aux = {}
        for name, decoder in self.decoders.items():
            output, dec_aux = decoder.decode(latent)
            outputs[name] = output
            decs_aux[name] = dec_aux

        return ReconstructionNetworkSubGameResult(
            encoder=self.encoder,
            decoders=self.decoders,
            input=input,
            latent=latent,
            outputs=outputs,
            encoder_aux=enc_aux,
            decoders_aux=decs_aux,
        )


class ReconstructionNetworkGame(Generic[T, U, AE, AD]):
    """
    A reconstruction network game composed of multiple sub-games.

    Args:
        agents (Mapping[str, IEncoderDecoder[T, U, AE, AD]]): A mapping of agent names to their respective IEncoderDecoder instances.
        adjacency (Dict[str, Set[str]] | None, optional): A dictionary representing the adjacency relationship between agents. Defaults to None.
    """

    def __init__(
        self,
        agents: Mapping[str, IEncoderDecoder[T, U, AE, AD]],
        adjacency: Dict[str, Set[str]] | None = None,
    ) -> None:
        self.agents = agents
        self.adjacency = adjacency or {}

        if set(self.adjacency) > set(agents):
            raise ValueError("Adjacency keys must be a subset of agent keys.")

        adj: Dict[str, Set[str]] = {k: set() for k in agents}
        adj.update(self.adjacency)
        self.adjacency = adj

        self.games = {
            name_e: ReconstructionNetworkSubGame(
                agent_e,
                {name_d: self.agents[name_d] for name_d in self.adjacency[name_e]},
            )
            for name_e, agent_e in agents.items()
        }

    def add_edge(self, source: str, target: str) -> None:
        """
        Add an edge between two agents in the game.

        Args:
            source (str): The source agent's name.
            target (str): The target agent's name.
        """
        self.adjacency[source].add(target)

    def remove_edge(self, source: str, target: str) -> None:
        """
        Remove an edge between two agents in the game.

        Args:
            source (str): The source agent's name.
            target (str): The target agent's name.
        """
        self.adjacency[source].remove(target)

    def __call__(
        self, input: T
    ) -> Dict[str, ReconstructionNetworkSubGameResult[T, U, AE, AD]]:
        """
        Execute the reconstruction network game on the given input data.

        Args:
            input (T): The input data.

        Returns:
            Dict[str, ReconstructionNetworkSubGameResult[T, U, AE, AD]]: A dictionary of sub-game results, keyed by agent names.
        """
        return {name: game(input) for name, game in self.games.items()}
