from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Mapping, Set, Tuple, TypeVar

T = TypeVar("T")
L = TypeVar("L")
M = TypeVar("M")
AOE = TypeVar("AOE")
AOD = TypeVar("AOD")
AME = TypeVar("AME")
AMD = TypeVar("AMD")


class IObjectEncoder(ABC, Generic[T, L, AOE]):
    @abstractmethod
    def encode_object(self, input: T) -> Tuple[L, AOE]:
        ...


class IObjectDecoder(ABC, Generic[L, T, M, AOD]):
    @abstractmethod
    def decode_object(
        self, latent: L, input: T | None = None, message: M | None = None
    ) -> Tuple[T, AOD]:
        ...


class IMessageEncoder(ABC, Generic[M, L, AME]):
    @abstractmethod
    def encode_message(self, message: M) -> Tuple[L, AME]:
        ...


class IMessageDecoder(ABC, Generic[L, M, T, AMD]):
    @abstractmethod
    def decode_message(
        self, latent: L, input: T | None = None, message: M | None = None
    ) -> Tuple[M, AMD]:
        ...


class INetworkAgent(
    IObjectEncoder[T, L, AOE],
    IObjectDecoder[L, T, M, AOD],
    IMessageEncoder[M, L, AME],
    IMessageDecoder[L, M, T, AMD],
    Generic[T, L, M, AOE, AOD, AME, AMD],
):
    ...


@dataclass
class ReconstructionNetworkSubGameResult(Generic[T, L, M, AOE, AOD, AME, AMD]):
    sender: INetworkAgent[T, L, M, AOE, AOD, AME, AMD]
    receivers: Mapping[str, INetworkAgent[T, L, M, AOE, AOD, AME, AMD]]
    input: T
    object_latent: L
    object_latent_aux: AOE
    message: M
    message_aux: AMD
    message_latents: Mapping[str, L]
    message_latents_aux: Mapping[str, AME]
    outputs: Mapping[str, T]
    outputs_aux: Mapping[str, AOD]


class ReconstructionNetworkSubGame(Generic[T, L, M, AOE, AOD, AME, AMD]):
    def __init__(
        self,
        sender: INetworkAgent[T, L, M, AOE, AOD, AME, AMD],
        receivers: Mapping[str, INetworkAgent[T, L, M, AOE, AOD, AME, AMD]],
    ) -> None:
        self.sender = sender
        self.receivers = receivers

    def __call__(
        self, input: T
    ) -> ReconstructionNetworkSubGameResult[T, L, M, AOE, AOD, AME, AMD]:
        object_latent, object_latent_aux = self.sender.encode_object(input)
        message, message_aux = self.sender.decode_message(object_latent, input=input)
        message_latents = {}
        message_latents_aux = {}
        outputs = {}
        outputs_aux = {}
        for name, receiver in self.receivers.items():
            message_latent, message_latent_aux = receiver.encode_message(message)
            output, output_aux = receiver.decode_object(message_latent, message=message)
            message_latents[name] = message_latent
            message_latents_aux[name] = message_latent_aux
            outputs[name] = output
            outputs_aux[name] = output_aux

        return ReconstructionNetworkSubGameResult(
            sender=self.sender,
            receivers=self.receivers,
            input=input,
            object_latent=object_latent,
            object_latent_aux=object_latent_aux,
            message=message,
            message_aux=message_aux,
            message_latents=message_latents,
            message_latents_aux=message_latents_aux,
            outputs=outputs,
            outputs_aux=outputs_aux,
        )


@dataclass
class ReconstructionNetworkGameResult(Generic[T, L, M, AOE, AOD, AME, AMD]):
    agents: Mapping[str, INetworkAgent[T, L, M, AOE, AOD, AME, AMD]]
    input: T
    subgame_results: Mapping[
        str, ReconstructionNetworkSubGameResult[T, L, M, AOE, AOD, AME, AMD]
    ]


class ReconstructionNetworkGame(Generic[T, L, M, AOE, AOD, AME, AMD]):
    def __init__(
        self,
        agents: Mapping[str, INetworkAgent[T, L, M, AOE, AOD, AME, AMD]],
        adjacency: Mapping[str, Set[str]] | None = None,
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

    def __call__(
        self, input: T
    ) -> ReconstructionNetworkGameResult[T, L, M, AOE, AOD, AME, AMD]:
        return ReconstructionNetworkGameResult(
            agents=self.agents,
            input=input,
            subgame_results={name: game(input) for name, game in self.games.items()},
        )

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
