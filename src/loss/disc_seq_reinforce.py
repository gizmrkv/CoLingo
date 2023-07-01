import torch
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType


class DiscSeqReinforceLoss(nn.Module):
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
        reward: TensorType["batch", float],
        log_prob: TensorType["batch", "length", float],
        entropy: TensorType["batch", "length", float] | None = None,
        length: TensorType["batch", int] | None = None,
    ):
        reward = reward.detach()
        log_prob = log_prob.sum(dim=-1)
        loss = -(reward - self.baseline(reward)) * log_prob

        if entropy is not None:
            ent_loss = self.entropy_weight * entropy.sum(dim=-1)
            loss -= ent_loss

        if length is not None:
            len_loss = self.length_weight * length.float()
            loss += (len_loss - self.length_baseline(len_loss)) * log_prob

        return loss
