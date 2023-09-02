from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

from ...core import Playable


@dataclass
class RecoSignalingGameResult:
    concept_encoder: nn.Module
    message_decoder: nn.Module
    message_encoder: nn.Module
    concept_decoder: nn.Module
    input: TensorType[..., int]
    input_feature: TensorType[..., float]
    message: TensorType[..., int]
    message_logits: TensorType[..., float]
    message_log_prob: TensorType[..., float]
    message_entropy: TensorType[..., float]
    message_length: TensorType[..., int]
    message_feature: TensorType[..., float]
    output: TensorType[..., int]
    output_logits: TensorType[..., float]


class RecoSignalingGame(Playable[TensorType[..., int], RecoSignalingGameResult]):
    def __init__(
        self,
        concept_encoder: nn.Module,
        message_decoder: nn.Module,
        message_encoder: nn.Module,
        concept_decoder: nn.Module,
    ) -> None:
        self.concept_encoder = concept_encoder
        self.message_decoder = message_decoder
        self.message_encoder = message_encoder
        self.concept_decoder = concept_decoder

    def play(
        self, input: TensorType[..., int], step: int | None = None
    ) -> RecoSignalingGameResult:
        input_feature = self.concept_encoder(input)
        message, message_logits = self.message_decoder(input_feature)

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

        message_feature = self.message_encoder(message)
        output, output_logits = self.concept_decoder(message_feature)

        return RecoSignalingGameResult(
            concept_encoder=self.concept_encoder,
            message_decoder=self.message_decoder,
            message_encoder=self.message_encoder,
            concept_decoder=self.concept_decoder,
            input=input,
            input_feature=input_feature,
            message=message,
            message_logits=message_logits,
            message_log_prob=log_prob,
            message_entropy=entropy,
            message_length=length,
            message_feature=message_feature,
            output=output,
            output_logits=output_logits,
        )
