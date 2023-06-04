import torch as th

from ..analysis import topographic_similarity
from .metric import Metric


class TopographicSimilarityMetric(Metric):
    def __init__(self, name: str = "topsim"):
        super().__init__(name=name)

    def calculate(
        self, input: th.Tensor, languages: dict[str, th.Tensor], *args, **kwds
    ):
        topsims = {}
        for agent_name, language in languages.items():
            topsims[agent_name] = topographic_similarity(
                input.numpy(), language.numpy()
            )

        return {self.name: topsims}
