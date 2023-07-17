import torch
from torch import nn


class BatchMeanBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.detach().mean(dim=0)


class MeanBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._count = 0

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._count += 1
            self._mean = (
                self._mean + (loss.detach().mean().item() - self._mean) / self._count
            )
        return self._mean
