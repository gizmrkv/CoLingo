import torch as th


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
