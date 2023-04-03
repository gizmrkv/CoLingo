import copy
import torch as th

from .baseline import build_baseline


def build_optimizer_type(optimizer_type: str):
    optimizers_dict = {"adam": th.optim.Adam}
    return optimizers_dict[optimizer_type]


class Agent(th.nn.Module):
    def __init__(
        self,
        model: th.nn.Module,
        optimizer_type: str,
        optimizer_args: dict,
        baseline_type: str,
        baseline_args: dict,
    ):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.optimizer = build_optimizer_type(optimizer_type)(
            self.model.parameters(), **optimizer_args
        )
        self.baseline = build_baseline(baseline_type, baseline_args)

    def forward(self, x: th.Tensor, input_type: str, is_training: bool = False):
        self.x, self.log_prob = self.model(x, input_type, is_training)
        return self.x

    def loss(self, reward: th.Tensor):
        self.baseline.update(self.x, reward)
        if self.log_prob:
            return -self.log_prob * (reward - self.baseline(self.x))
        else:
            return -reward
