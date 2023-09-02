from typing import Callable

from torch import nn
from torchtyping import TensorType


class ReinforceLoss(nn.Module):
    """
    Custom loss function for reinforcement learning using the REINFORCE algorithm.
    """

    def __init__(
        self,
        max_len: int,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
        length_baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
    ) -> None:
        """
        Initialize the ReinforceLoss instance.

        Args:
            max_len (int): Maximum sequence length.
            entropy_weight (float, optional): Weight for entropy regularization term. Default is 0.0 (no regularization).
            length_weight (float, optional): Weight for length-related regularization term. Default is 0.0 (no regularization).
            baseline (Callable[[TensorType[..., float]], TensorType[..., float]] | None, optional):
                Baseline function for reward. Default is None.
            length_baseline (Callable[[TensorType[..., float]], TensorType[..., float]] | None, optional):
                Baseline function for length-related regularization. Default is None.
        """
        super().__init__()
        self.max_len = max_len
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline

    def forward(
        self,
        reward: TensorType[..., float],
        log_prob: TensorType[..., "max_len", float],
        entropy: TensorType[..., "max_len", float] | None = None,
        length: TensorType[..., int] | None = None,
    ) -> TensorType[..., float]:
        """
        Compute the reinforcement loss.

        Args:
            reward (TensorType[..., float]): Reward tensor.
            log_prob (TensorType[..., "max_len", float]): Log probability tensor for actions.
            entropy (TensorType[..., "max_len", float] | None, optional): Entropy tensor. Default is None.
            length (TensorType[..., int] | None, optional): Sequence length tensor. Default is None.

        Returns:
            TensorType[..., float]: Computed loss tensor.
        """

        reward = reward.detach()
        log_prob = log_prob.sum(dim=-1)

        if self.baseline is not None:
            reward -= self.baseline(reward)
        loss = -reward * log_prob

        if entropy is not None:
            ent_loss = self.entropy_weight * entropy.sum(dim=-1)
            loss -= ent_loss

        if length is not None:
            len_loss = self.length_weight * length.float() / self.max_len
            if self.length_baseline is not None:
                len_loss -= self.length_baseline(len_loss)
            loss += len_loss * log_prob

        return loss
