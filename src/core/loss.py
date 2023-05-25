import torch as th
from torchtyping import TensorType


class ReinforceLoss(th.nn.Module):
    """
    This class implements a customized loss function based on the REINFORCE algorithm.
    This loss function is used to train agents in reinforcement learning settings.

    Args:
        entropy_weight (float, optional): The weight applied to the entropy term in the loss function. Defaults to 0.0.
        length_weight (float, optional): The weight applied to the length term in the loss function. Defaults to 0.0.
        baseline (th.nn.Module | None, optional): A module to compute the baseline for variance reduction. Defaults to None.
        length_baseline (th.nn.Module | None, optional): A module to compute the baseline for the length term. Defaults to None.
    """

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
        self,
        loss: TensorType["batch", float],
        logprob: TensorType["batch", float],
        entropy: TensorType["batch", float],
        length: TensorType["batch", int],
    ) -> TensorType["batch", float]:
        """
        Compute the REINFORCE loss.

        Args:
            loss (TensorType["batch", float]): The original loss values.
            auxiliary (dict): A dictionary that includes 'logprob', 'entropy', and 'length' as keys.

        Returns:
            TensorType["batch", float]: The computed REINFORCE loss.
        """
        loss = loss.detach()
        length = length.float() * self.length_weight

        policy_loss = (loss - self.baseline(loss)) * logprob
        entropy = self.entropy_weight * entropy
        length_loss = (length - self.length_baseline(length)) * logprob
        return policy_loss - entropy + length_loss


class ConceptLoss(th.nn.Module):
    """
    The ConceptLoss class implements a loss function.

    Args:
        n_attributes (int): The number of attributes in the concept.
        n_values (int): The number of values each attribute can take.
    """

    def __init__(self, n_attributes: int, n_values: int):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(
        self,
        input: TensorType["batch", "n_attributes", "n_values", float],
        target: th.Tensor,
    ) -> TensorType["batch", float]:
        """
        Compute the Concept loss.

        Args:
            input (TensorType["batch", "n_attributes", "n_values", float]): The prediction from the model.
            target (th.Tensor): The ground truth.

        Returns:
            TensorType["batch", float]: The computed Concept loss.
        """
        input = input.view(-1, self.n_values)
        target = target.view(-1)
        return (
            th.nn.functional.cross_entropy(input, target, reduction="none")
            .view(-1, self.n_attributes)
            .mean(dim=-1)
        )


class OnehotConceptLoss(th.nn.Module):
    def __init__(self, n_attributes: int, n_values: int):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, input: th.Tensor, target: th.Tensor):
        bsz = input.shape[0]
        input = input.view(bsz * self.n_attributes, self.n_values)
        target = (
            target.view(bsz, self.n_attributes, self.n_values)
            .argmax(dim=-1)
            .view(bsz * self.n_attributes)
        )
        target = target.view(bsz * self.n_attributes)
        return (
            th.nn.functional.cross_entropy(input, target, reduction="none")
            .view(-1, self.n_attributes)
            .mean(dim=-1)
        )
