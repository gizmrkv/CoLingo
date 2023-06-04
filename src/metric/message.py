import torch as th

from .metric import Metric


class MessageMetric(Metric):
    def __init__(self, name: str = "message"):
        super().__init__(name=name)

    def calculate(
        self,
        message: th.Tensor,
        log_prob: th.Tensor,
        entropy: th.Tensor,
        length: th.Tensor,
        *args,
        **kwds,
    ):
        n = message.shape[0]
        message = message.sort(dim=1)[0]
        message = th.unique(message, dim=0)
        uniques = message.shape[0] / n
        return {
            self.name: {
                "log_prob": log_prob.mean().item(),
                "entropy": entropy.mean().item(),
                "length": length.float().mean().item(),
                "uniques": uniques,
            }
        }
