from typing import Dict, Iterable, Mapping

from ...core import Loggable
from .game import RecoSignalingGameResult
from .loss import Loss


class MetricsLogger(Loggable[RecoSignalingGameResult]):
    def __init__(
        self, loss: Loss, loggers: Iterable[Loggable[Mapping[str, float]]]
    ) -> None:
        self.loss = loss
        self.loggers = loggers

    def log(self, result: RecoSignalingGameResult, step: int | None = None) -> None:
        metrics: Dict[str, float] = {}

        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics |= {
            "entropy": result.message_entropy.mean().item(),
            "length": result.message_length.float().mean().item(),
            "unique": result.message.unique(dim=0).shape[0] / result.message.shape[0],
        }

        loss_r = self.loss.receiver_loss(result)
        loss_s = self.loss.sender_loss(result, loss_r)
        total_loss = loss_r + loss_s
        metrics |= {
            "decoder_loss": loss_r.mean().item(),
            "encoder_loss": loss_s.mean().item(),
            "total_loss": total_loss.mean().item(),
        }

        for logger in self.loggers:
            logger.log(metrics)
