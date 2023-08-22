from typing import Iterable

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from ...game import ReconstructionGameResult


def loss(
    step: int,
    input: TensorType[..., int],
    outputs: Iterable[
        ReconstructionGameResult[
            TensorType[..., int], TensorType[..., float], None, TensorType[..., float]
        ]
    ],
) -> TensorType[1, float]:
    return torch.stack(
        [
            F.cross_entropy(
                output.decoder_aux.view(-1, output.decoder_aux.shape[-1]),
                output.input.view(-1),
            )
            for output in outputs
        ]
    ).mean()
