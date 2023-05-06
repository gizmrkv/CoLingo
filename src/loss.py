import torch as th


class ReinforceLoss(th.nn.Module):
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

    def forward(self, loss: th.Tensor, auxiliary: dict):
        loss = loss.detach()
        logprob = auxiliary["logprob"]
        entropy = auxiliary["entropy"]
        length = auxiliary["length"].float() * self.length_weight

        policy_loss = (loss - self.baseline(loss)) * logprob
        entropy = self.entropy_weight * entropy
        length_loss = (length - self.length_baseline(length)) * logprob
        return policy_loss - entropy + length_loss


class ConceptLoss(th.nn.Module):
    def __init__(self, n_attributes: int, n_values: int):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, input: th.Tensor, target: th.Tensor):
        bsz = input.shape[0]
        input = input.view(bsz * self.n_attributes, self.n_values)
        target = target.view(bsz * self.n_attributes)
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
