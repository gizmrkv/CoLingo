import torch.nn.functional as F
from torchtyping import TensorType

from .game import ReconstructionGameResult


class Loss:
    def __call__(
        self,
        result: ReconstructionGameResult,
    ) -> TensorType[1, float]:
        return F.cross_entropy(
            result.logits.view(-1, result.logits.shape[-1]),
            result.input.view(-1),
            reduction="mean",
        )
