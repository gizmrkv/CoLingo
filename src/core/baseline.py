import torch as th


class MeanBaseline(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._mean = th.nn.Parameter(th.zeros(1), requires_grad=False)
        self._count = 0

    def forward(self, loss: th.Tensor) -> th.Tensor:
        if self.training:
            self._count += 1
            self._mean += (loss.mean().item() - self._mean) / self._count
        return self._mean


class BatchMeanBaseline(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, loss: th.Tensor) -> th.Tensor:
        return loss.mean(dim=0).detach()
