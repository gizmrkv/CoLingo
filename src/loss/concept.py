import torch as th
from torchtyping import TensorType


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
