from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from torchtyping import TensorType

from ...analysis import topographic_similarity
from ...core import Computable, Loggable
from ...loggers import LanguageLogger
from .game import RecoSignalingGameResult
from .loss import Loss


class Metrics(Computable[TensorType[..., int], RecoSignalingGameResult, None]):
    def __init__(
        self, loss: Loss, loggers: Iterable[Loggable[Mapping[str, float]]]
    ) -> None:
        self.loss = loss
        self.loggers = loggers

    def compute(
        self,
        input: TensorType[..., int],
        output: RecoSignalingGameResult,
        step: int | None = None,
    ) -> None:
        metrics: Dict[str, float] = {}

        mark = output.output == output.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics |= {
            "entropy": output.message_entropy.mean().item(),
            "length": output.message_length.float().mean().item(),
            "unique": output.message.unique(dim=0).shape[0] / output.message.shape[0],
        }

        loss_r = self.loss.receiver_loss(output)
        loss_s = self.loss.sender_loss(output, loss_r)
        total_loss = loss_r + loss_s
        metrics |= {
            "decoder_loss": loss_r.mean().item(),
            "encoder_loss": loss_s.mean().item(),
            "total_loss": total_loss.mean().item(),
        }

        for logger in self.loggers:
            logger.log(metrics)


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]


class TopographicSimilarityMetrics(
    Computable[TensorType[..., int], RecoSignalingGameResult, None]
):
    def __init__(self, loggers: Iterable[Loggable[Mapping[str, float]]]) -> None:
        self.loggers = loggers

    def compute(
        self,
        input: TensorType[..., int],
        output: RecoSignalingGameResult,
        step: int | None = None,
    ) -> None:
        metrics = {
            f"topsim": topographic_similarity(
                output.input.cpu().numpy(),
                output.message.cpu().numpy(),
                y_processor=drop_padding,  # type: ignore
            )
        }
        for logger in self.loggers:
            logger.log(metrics)


class LanguageLoggerWrapper(
    Computable[TensorType[..., int], RecoSignalingGameResult, None]
):
    def __init__(self, save_dir: Path) -> None:
        self.logger = LanguageLogger(save_dir)

    def compute(
        self,
        input: TensorType[..., int],
        output: RecoSignalingGameResult,
        step: int | None = None,
    ) -> None:
        self.logger.log((step or 0, input, output.message))
