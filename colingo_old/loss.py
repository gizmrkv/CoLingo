from torch import nn
from torchtyping import TensorType


class ReinforceLoss(nn.Module):
    BATCH = "batch"
    LENGTH = "length"

    def __init__(
        self,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        baseline: nn.Module | None = None,
        length_baseline: nn.Module | None = None,
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline

    def forward(
        self,
        reward: TensorType[BATCH, float],
        log_prob: TensorType[BATCH, LENGTH, float],
        entropy: TensorType[BATCH, LENGTH, float] | None = None,
        length: TensorType[BATCH, int] | None = None,
    ) -> TensorType[BATCH, float]:
        reward = reward.detach()
        log_prob = log_prob.sum(dim=-1)

        if self.baseline is not None:
            reward -= self.baseline(reward)
        loss = -reward * log_prob

        if entropy is not None:
            ent_loss = self.entropy_weight * entropy.sum(dim=-1)
            loss -= ent_loss

        if length is not None:
            len_loss = self.length_weight * length.float()
            if self.length_baseline is not None:
                len_loss -= self.length_baseline(len_loss)
            loss += len_loss * log_prob

        return loss
