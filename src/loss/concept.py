import torch as th
from torchtyping import TensorType


class ConceptLoss(th.nn.Module):
    def __init__(self, n_attributes: int, n_values: int):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(
        self,
        input: TensorType["batch", "n_attributes", "n_values", float],
        target: th.Tensor,
    ) -> TensorType["batch", float]:
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
