import torch as th

from .metric import Metric


class ConceptAccuracyMetric(Metric):
    def __init__(
        self, n_attributes: int, n_values: int, name: str = "concept_accuracy"
    ):
        super().__init__(name=name)
        self.n_attributes = n_attributes
        self.n_values = n_values

    def calculate(self, output: th.Tensor, target: th.Tensor, *args, **kwargs):
        output = output.argmax(dim=-1)
        acc = {}
        acc["partial"] = (output == target).float().mean().item()
        acc["complete"] = (output == target).all(dim=-1).float().mean().item()
        for i in range(self.n_attributes):
            acc[i + 1] = (output[:, i] == target[:, i]).float().mean().item()

        return {self.name: acc}
