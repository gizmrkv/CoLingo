import torch as th


class MeanBaseline(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = th.nn.Parameter(th.zeros(1), requires_grad=False)
        self.count = 0

    def forward(self, loss: th.Tensor) -> th.Tensor:
        if self.training:
            self.count += 1
            self.mean += (loss.mean().item() - self.mean) / self.count
        return self.mean


class BatchMeanBaseline(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, loss: th.Tensor) -> th.Tensor:
        return loss.mean(dim=0).detach()
