from typing import Callable, Dict, Iterable

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from ...game import ReconstructionNetworkSubGameResult
from ...loss import ReinforceLoss
from .agent import MessageAuxiliary


class Loss:
    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        self.object_length = object_length
        self.object_n_values = object_n_values
        self.message_length = message_length
        self.message_n_values = message_n_values
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline

        self.reinforce_loss = ReinforceLoss(
            max_len=message_length,
            entropy_weight=entropy_weight,
            length_weight=length_weight,
            baseline=baseline,
            length_baseline=baseline,
        )

    def decoders_loss(
        self,
        result: ReconstructionNetworkSubGameResult[
            TensorType[..., int],
            TensorType[..., int],
            MessageAuxiliary,
            TensorType[..., float],
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
            TensorType[..., int],
            TensorType[..., int],
            MessageAuxiliary,
            TensorType[..., float],
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
            TensorType[..., int],
            TensorType[..., int],
            MessageAuxiliary,
            TensorType[..., float],
        ],
    ) -> TensorType[..., float]:
        decoders_loss = self.decoders_loss(result)
        decoder_loss = torch.stack(list(decoders_loss.values()), dim=-1).mean(dim=-1)
        encoder_loss = self.encoder_loss(result, decoder_loss)
        return decoder_loss + encoder_loss

    def total_loss(
        self,
        output: Dict[
            str,
            ReconstructionNetworkSubGameResult[
                TensorType[..., int],
                TensorType[..., int],
                MessageAuxiliary,
                TensorType[..., float],
            ],
        ],
    ) -> TensorType[..., float]:
        return torch.stack(
            [self.sub_loss(result) for result in output.values()], dim=-1
        ).mean(dim=-1)

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., int],
                    TensorType[..., int],
                    MessageAuxiliary,
                    TensorType[..., float],
                ],
            ]
        ],
    ) -> TensorType[1, float]:
        return torch.stack([self.total_loss(output) for output in outputs]).mean()
