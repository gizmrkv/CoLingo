from typing import Dict, Iterable, Mapping

from torch.distributions import Categorical

from ...core import Loggable
from .game import ReconstructionGameResult
from .loss import Loss


class MetricsLogger(Loggable[ReconstructionGameResult]):
    def __init__(
        self, loss: Loss, loggers: Iterable[Loggable[Mapping[str, float]]]
    ) -> None:
        self.loggers = loggers
        self.loss = loss

    def log(self, result: ReconstructionGameResult, step: int | None = None) -> None:
        metrics: Dict[str, float] = {}
        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics["entropy"] = Categorical(logits=result.logits).entropy().mean().item()

        metrics["loss"] = self.loss(result).mean().item()

        for logger in self.loggers:
            logger.log(metrics)
