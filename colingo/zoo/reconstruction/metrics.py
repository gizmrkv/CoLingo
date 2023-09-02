from typing import Dict, Iterable, Mapping

from torch.distributions import Categorical
from torchtyping import TensorType

from ...core import Computable, Loggable
from .game import ReconstructionGameResult
from .loss import Loss


class Metrics(Computable[TensorType[..., int], ReconstructionGameResult, None]):
    def __init__(
        self, loss: Loss, loggers: Iterable[Loggable[Mapping[str, float]]]
    ) -> None:
        self.loggers = loggers
        self.loss = loss

    def compute(
        self,
        input: TensorType[..., int],
        output: ReconstructionGameResult,
        step: int | None = None,
    ) -> None:
        metrics: Dict[str, float] = {}
        mark = output.output == output.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics["entropy"] = Categorical(logits=output.logits).entropy().mean().item()

        metrics["loss"] = self.loss.compute(input, output).mean().item()

        for logger in self.loggers:
            logger.log(metrics)
