import torch as th


class MeanBaseline(th.nn.Module):
    """
    This class represents a baseline model that calculates and keeps track of the
    mean of the loss function during the training phase.

    Methods:
        forward(loss: th.Tensor): This method calculates the mean of the loss tensor
            during the training phase and returns it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mean = th.nn.Parameter(th.zeros(1), requires_grad=False)
        self._count = 0

    def forward(self, loss: th.Tensor) -> th.Tensor:
        """
        If the model is in training mode, this method calculates the mean of the loss
        tensor and updates the current mean value accordingly. It then returns the current
        mean value.

        Args:
            loss (th.Tensor): The loss tensor whose mean value is to be calculated.

        Returns:
            th.Tensor: The current mean value of the loss tensor.
        """

        if self.training:
            self._count += 1
            self._mean += (loss.mean().item() - self._mean) / self._count
        return self._mean


class BatchMeanBaseline(th.nn.Module):
    """
    This class represents a baseline model that calculates the mean of the loss function
    across the batch dimension and detaches it from the computation graph.

    Methods:
        forward(loss: th.Tensor): This method calculates the mean of the loss tensor
            across the batch dimension, detaches it from the computation graph, and returns it.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, loss: th.Tensor) -> th.Tensor:
        """
        This method calculates the mean of the loss tensor across the batch dimension,
        detaches it from the computation graph, and returns it. The resulting tensor has
        no gradient.

        Args:
            loss (th.Tensor): The loss tensor whose mean value across the batch dimension
                is to be calculated.

        Returns:
            th.Tensor: The mean value of the loss tensor across the batch dimension, detached
                from the computation graph.
        """
        return loss.mean(dim=0).detach()
