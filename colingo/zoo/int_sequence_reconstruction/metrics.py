from typing import Callable, Iterable

from torchtyping import TensorType

from ...game import ReconstructionGameResult
from .loss import loss


class Metrics:
    def __init__(
        self,
        name: str,
        length: int,
        n_values: int,
        callbacks: Iterable[Callable[[dict[str, float]], None]],
    ) -> None:
        self.name = name
        self.length = length
        self.n_values = n_values
        self.callbacks = callbacks

    def __call__(
        self,
        step: int,
        input: TensorType[..., "length", int],
        outputs: Iterable[
            ReconstructionGameResult[
                TensorType[..., "length", int],
                TensorType[..., float],
                None,
                TensorType[..., "length", "n_values", float],
            ]
        ],
    ) -> None:
        metrics: dict[str, float] = {}

        output = next(iter(outputs))

        mark = output.output == output.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics["loss"] = loss(step, input, outputs).mean().item()

        metrics = {f"{self.name}.{k}": v for k, v in metrics.items()}
        for callback in self.callbacks:
            callback(metrics)
