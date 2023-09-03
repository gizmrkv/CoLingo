from dataclasses import dataclass
from typing import Dict, Mapping, Set

import torch
from networkx import DiGraph
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

from ...core import Playable


class RecoNetworkAgent(nn.Module):
    def __init__(
        self,
        concept_encoder: nn.Module,
        message_decoder: nn.Module,
        message_encoder: nn.Module,
        concept_decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.concept_encoder = concept_encoder
        self.message_decoder = message_decoder
        self.message_encoder = message_encoder
        self.concept_decoder = concept_decoder


@dataclass
class RecoNetworkSubGameResult:
    sender: RecoNetworkAgent
    receivers: Mapping[str, RecoNetworkAgent]
    input: TensorType[..., int]
    input_feature: TensorType[..., float]
    message: TensorType[..., int]
    message_logits: TensorType[..., float]
    message_log_prob: TensorType[..., float]
    message_entropy: TensorType[..., float]
    message_length: TensorType[..., int]
    message_features: Mapping[str, TensorType[..., float]]
    outputs: Mapping[str, TensorType[..., int]]
    outputs_logits: Mapping[str, TensorType[..., float]]


class RecoNetworkSubGame(Playable[TensorType[..., int], RecoNetworkSubGameResult]):
    def __init__(
        self,
        sender: RecoNetworkAgent,
        receivers: Mapping[str, RecoNetworkAgent],
    ) -> None:
        self.sender = sender
        self.receivers = receivers

    def play(
        self, input: TensorType[..., int], step: int | None = None
    ) -> RecoNetworkSubGameResult:
        input_feature = self.sender.concept_encoder(input)
        message, message_logits = self.sender.message_decoder(
            input_feature, concept=input
        )

        distr = Categorical(logits=message_logits)
        log_prob = distr.log_prob(message)
        entropy = distr.entropy()

        mask = message == 0
        indices = torch.argmax(mask.int(), dim=1)
        no_mask = ~mask.any(dim=1)
        indices[no_mask] = message.shape[1]
        mask = torch.arange(message.shape[1]).expand(message.shape).to(message.device)
        mask = (mask <= indices.unsqueeze(-1)).long()

        length = mask.sum(dim=-1)
        message = message * mask
        log_prob = log_prob * mask
        entropy = entropy * mask

        message_features = {}
        outputs = {}
        outputs_logits = {}

        for name, receiver in self.receivers.items():
            message_feature = receiver.message_encoder(message)
            output, output_logits = receiver.concept_decoder(
                message_feature, message=message
            )
            message_features[name] = message_feature
            outputs[name] = output
            outputs_logits[name] = output_logits

        return RecoNetworkSubGameResult(
            sender=self.sender,
            receivers=self.receivers,
            input=input,
            input_feature=input_feature,
            message=message,
            message_logits=message_logits,
            message_log_prob=log_prob,
            message_entropy=entropy,
            message_length=length,
            message_features=message_features,
            outputs=outputs,
            outputs_logits=outputs_logits,
        )


@dataclass
class RecoNetworkGameResult:
    agents: Mapping[str, RecoNetworkAgent]
    network: DiGraph
    input: TensorType[..., int]
    sub_results: Mapping[str, RecoNetworkSubGameResult]


class RecoNetworkGame(Playable[TensorType[..., int], RecoNetworkGameResult]):
    def __init__(
        self, agents: Mapping[str, RecoNetworkAgent], network: DiGraph
    ) -> None:
        self.agents = agents
        self.network = network

        self.games = {
            name_s: RecoNetworkSubGame(
                agent_s,
                {name_r: self.agents[name_r] for name_r in self.network.succ[name_s]},
            )
            for name_s, agent_s in agents.items()
        }

    def play(
        self, input: TensorType[..., int], step: int | None = None
    ) -> RecoNetworkGameResult:
        return RecoNetworkGameResult(
            agents=self.agents,
            network=self.network,
            input=input,
            sub_results={
                name: game.play(input, step=step) for name, game in self.games.items()
            },
        )
