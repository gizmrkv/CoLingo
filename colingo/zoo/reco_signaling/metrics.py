from typing import Callable, Dict, Iterable

import numpy as np
from numpy.typing import NDArray
from torchtyping import TensorType

from ...analysis import topographic_similarity
from ...game import ReconstructionGameResult
from ...loggers import IntSequenceLanguageLogger
from .agent import MessageAuxiliary
from .loss import Loss


class Metrics:
    def __init__(
        self,
        name: str,
        loss: Loss,
        callbacks: Iterable[Callable[[Dict[str, float]], None]],
    ) -> None:
        self.name = name
        self.loss = loss
        self.callbacks = callbacks

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            ReconstructionGameResult[
                TensorType[..., int],
                TensorType[..., int],
                MessageAuxiliary,
                TensorType[..., float],
            ]
        ],
    ) -> None:
        metrics: Dict[str, float] = {}

        output = next(iter(outputs))

        mark = output.output == output.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics |= {
            "entropy": output.encoder_aux.entropy.mean().item(),
            "length": output.encoder_aux.length.float().mean().item(),
            "unique": output.encoder_aux.message.unique(dim=0).shape[0]
            / output.encoder_aux.message.shape[0],
        }

        decoder_loss = self.loss.decoder_loss(output)
        encoder_loss = self.loss.encoder_loss(output, decoder_loss)
        total_loss = decoder_loss + encoder_loss
        metrics |= {
            "decoder_loss": decoder_loss.mean().item(),
            "encoder_loss": encoder_loss.mean().item(),
            "total_loss": total_loss.mean().item(),
        }

        metrics = {f"{self.name}.{k}": v for k, v in metrics.items()}
        for callback in self.callbacks:
            callback(metrics)


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]


class TopographicSimilarityMetrics:
    def __init__(
        self, name: str, callbacks: Iterable[Callable[[Dict[str, float]], None]]
    ) -> None:
        self.name = name
        self.callbacks = callbacks

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            ReconstructionGameResult[
                TensorType[..., int],
                TensorType[..., int],
                MessageAuxiliary,
                TensorType[..., float],
            ]
        ],
    ) -> None:
        output = next(iter(outputs))
        metrics = {
            f"{self.name}.topsim": topographic_similarity(
                output.input.cpu().numpy(),
                output.latent.cpu().numpy(),
                y_processor=drop_padding,  # type: ignore
            )
        }

        for callback in self.callbacks:
            callback(metrics)


class LanguageLogger:
    def __init__(self, save_dir: str) -> None:
        self.logger = IntSequenceLanguageLogger(save_dir)

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            ReconstructionGameResult[
                TensorType[..., int],
                TensorType[..., int],
                MessageAuxiliary,
                TensorType[..., float],
            ]
        ],
    ) -> None:
        output = next(iter(outputs))
        self.logger(step, output.input, output.latent)
