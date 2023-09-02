import torch.nn.functional as F
from torchtyping import TensorType

from ...core import Computable
from .game import ReconstructionGameResult


class Loss(
    Computable[TensorType[..., int], ReconstructionGameResult, TensorType[1, float]]
):
    def compute(
        self,
        input: TensorType[..., int],
        output: ReconstructionGameResult,
        step: int | None = None,
    ) -> TensorType[1, float]:
        return F.cross_entropy(
            output.logits.view(-1, output.logits.shape[-1]),
            input.view(-1),
            reduction="mean",
        )
