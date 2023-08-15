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
    encoder: IEncoder[T, U, AE]
    decoders: Dict[str, IDecoder[U, T, AD]]
    input: T
    latent: U
    outputs: Dict[str, T]
    encoder_aux: AE
    decoders_aux: Dict[str, AD]


class ReconstructionNetworkSubGame(Generic[T, U, AE, AD]):
    def __init__(
        self, encoder: IEncoder[T, U, AE], decoders: Dict[str, IDecoder[U, T, AD]]
    ) -> None:
        self.encoder = encoder
        self.decoders = decoders

    def __call__(self, input: T) -> ReconstructionNetworkSubGameResult[T, U, AE, AD]:
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
        self.adjacency[source].add(target)

    def remove_edge(self, source: str, target: str) -> None:
        self.adjacency[source].remove(target)

    def __call__(
        self, input: T
    ) -> Dict[str, ReconstructionNetworkSubGameResult[T, U, AE, AD]]:
        return {name: game(input) for name, game in self.games.items()}
