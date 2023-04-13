import copy
import torch as th

from . import baseline


def build_optimizer_type(optimizer_type: str):
    optimizers_dict = {"adam": th.optim.Adam}
    return optimizers_dict[optimizer_type]


class Agent(th.nn.Module):
    def __init__(
        self,
        model: th.nn.Module,
        baseline: baseline.Baseline,
        optimizer_type: str,
        optimizer_args: dict,
    ):
        super().__init__()
        self.model = copy.deepcopy(model)
        for param in self.model.parameters():
            th.nn.init.normal_(param)

        self.baseline = copy.deepcopy(baseline)
        self.optimizer = build_optimizer_type(optimizer_type)(
            self.model.parameters(), **optimizer_args
        )

    def forward(self, x: th.Tensor, input_type: str):
        self.x, self.log_prob, self.entropy = self.model(x, input_type)
        return self.x

    def loss(self, reward: th.Tensor):
        self.baseline.update(self.x, reward)
        if self.log_prob is not None:
            return -self.log_prob.mean() * (
                reward - self.baseline(self.x) - 0.0 * self.entropy.mean()
            )
        else:
            return -reward
