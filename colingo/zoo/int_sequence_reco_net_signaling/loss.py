from typing import Callable, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from ...game import ReconstructionNetworkSubGameResult
from ...loss import ReinforceLoss
from .agent import Agent, MessageAuxiliary


class Loss:
    def __init__(
        self,
        agents: Mapping[str, Agent],
        object_length: int,
        object_n_values: int,
        message_length: int,
        message_n_values: int,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
        length_baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
        decoder_ae: bool = False,
    ) -> None:
        super().__init__()
        self.object_length = object_length
        self.object_n_values = object_n_values
        self.message_length = message_length
        self.message_n_values = message_n_values
        self.agents = agents
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline
        self.decoder_ae = decoder_ae

        self.reinforce_loss = ReinforceLoss(
            max_len=message_length,
            entropy_weight=entropy_weight,
            length_weight=length_weight,
            baseline=baseline,
            length_baseline=baseline,
        )

    def __call__(
        self,
        step: int,
        input: TensorType[..., "object_length", int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., "object_length", int],
                    TensorType[..., "message_length", int],
                    MessageAuxiliary,
                    TensorType[..., "object_length", "object_n_values", float],
                ],
            ]
        ],
    ) -> TensorType[1, float]:
        return torch.stack([self.total_loss(output) for output in outputs]).mean()

    def decoders_loss(
        self,
        result: ReconstructionNetworkSubGameResult[
            TensorType[..., "object_length", int],
            TensorType[..., "message_length", int],
            MessageAuxiliary,
            TensorType[..., "object_length", "object_n_values", float],
        ],
    ) -> Dict[str, TensorType[..., float]]:
        return {
            name: F.cross_entropy(
                logits.view(-1, self.object_n_values),
                result.input.view(-1),
                reduction="none",
            )
            .view(-1, self.object_length)
            .mean(dim=-1)
            for name, logits in result.decoders_aux.items()
        }

    def encoder_loss(
        self,
        result: ReconstructionNetworkSubGameResult[
            TensorType[..., "object_length", int],
            TensorType[..., "message_length", int],
            MessageAuxiliary,
            TensorType[..., "object_length", "object_n_values", float],
        ],
        decoder_loss: TensorType[..., float],
    ) -> TensorType[..., float]:
        return self.reinforce_loss(
            reward=-decoder_loss.detach(),
            log_prob=result.encoder_aux.log_prob,
            entropy=result.encoder_aux.entropy,
            length=result.encoder_aux.length,
        )

    def sub_loss(
        self,
        result: ReconstructionNetworkSubGameResult[
            TensorType[..., "object_length", int],
            TensorType[..., "message_length", int],
            MessageAuxiliary,
            TensorType[..., "object_length", "object_n_values", float],
        ],
    ) -> TensorType[..., float]:
        if len(result.decoders) == 0:
            return torch.zeros(
                result.input.size(0), device=result.input.device, dtype=torch.float32
            )

        decoders_loss = self.decoders_loss(result)
        decoder_loss = torch.stack(list(decoders_loss.values()), dim=-1).mean(dim=-1)
        encoder_loss = self.encoder_loss(result, decoder_loss)
        loss = decoder_loss + encoder_loss

        if self.decoder_ae:
            loss += self.decoder_ae_loss(result)

        return loss

    def total_loss(
        self,
        output: Dict[
            str,
            ReconstructionNetworkSubGameResult[
                TensorType[..., "object_length", int],
                TensorType[..., "message_length", int],
                MessageAuxiliary,
                TensorType[..., "object_length", "object_n_values", float],
            ],
        ],
    ) -> TensorType[..., float]:
        return torch.stack(
            [self.sub_loss(result) for result in output.values()], dim=-1
        ).mean(dim=-1)

    def decoder_ae_loss(
        self,
        result: ReconstructionNetworkSubGameResult[
            TensorType[..., "object_length", int],
            TensorType[..., "message_length", int],
            MessageAuxiliary,
            TensorType[..., "object_length", "object_n_values", float],
        ],
    ) -> TensorType[..., float]:
        agents = {k: self.agents[k] for k in result.decoders}
        logits = {
            k: agent.encode(result.outputs[k])[1].logits for k, agent in agents.items()
        }
        masks = {
            k: (output == result.input).all(dim=-1).float()
            for k, output in result.outputs.items()
        }
        losses = [
            mask
            * F.cross_entropy(
                logit.view(-1, self.message_n_values),
                result.latent.view(-1),
                reduction="none",
            )
            .view(-1, self.message_length)
            .mean(dim=-1)
            for logit, mask in zip(logits.values(), masks.values())
        ]
        return torch.stack(losses, dim=-1).mean(dim=-1)
