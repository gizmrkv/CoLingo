import torch as th
from abc import ABC, abstractmethod


class Baseline(ABC):
    @abstractmethod
    def __call__(self, state: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def update(self, state: th.Tensor, reward: th.Tensor):
        raise NotImplementedError


class MeanBaseline(Baseline):
    def __init__(self) -> None:
        self.mean = th.zeros(1, requires_grad=False)
        self.count = 0

    def __call__(self, state: th.Tensor) -> th.Tensor:
        return self.mean

    def update(self, state: th.Tensor, reward: th.Tensor):
        self.count += 1
        self.mean += (reward.detach().mean().item() - self.mean) / self.count


def build_baseline(baseline_type: str, baseline_args: dict) -> Baseline:
    baselines_dict = {"mean": MeanBaseline}
    return baselines_dict[baseline_type](**baseline_args)
