import torch as th
from torchtyping import TensorType

from ..message import SequenceMessage


class SequenceMessageLoss(th.nn.Module):
    def __init__(
        self,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        baseline: th.nn.Module | None = None,
        length_baseline: th.nn.Module | None = None,
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline

    def forward(
        self, sequence: SequenceMessage, loss: TensorType["batch", float]
    ) -> TensorType["batch", float]:
        log_prob = sequence.log_prob
        entropy = sequence.entropy
        length = sequence.length

        loss = loss.detach()
        length = length.float() * self.length_weight

        policy_loss = (loss - self.baseline(loss)) * log_prob
        entropy = self.entropy_weight * entropy
        length_loss = (length - self.length_baseline(length)) * log_prob
        return policy_loss - entropy + length_loss
