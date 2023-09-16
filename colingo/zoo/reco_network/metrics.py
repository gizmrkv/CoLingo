from statistics import fmean
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import torch
from torchtyping import TensorType

from ...core import Loggable
from ...loggers import HeatmapLogger
from .game import RecoNetworkGameResult
from .loss import Loss


class MetricsLogger(Loggable[RecoNetworkGameResult]):
    def __init__(
        self, loss: Loss, loggers: Iterable[Loggable[Mapping[str, float]]]
    ) -> None:
        self.loss = loss
        self.loggers = loggers

    def log(self, output: RecoNetworkGameResult, step: int | None = None) -> None:
        agent_metrics: Dict[str, Dict[str, float]] = {}

        for name_s, result_s in output.sub_results.items():
            acc_comps, acc_parts = zip(
                *[
                    acc_comp_part(output_r, result_s.input)
                    for output_r in result_s.outputs.values()
                ]
            )
            loss_rs = self.loss.receivers_loss(result_s)
            loss_r = torch.stack(list(loss_rs.values()), dim=-1).mean(dim=-1)
            loss_s = self.loss.sender_loss(result_s, loss_r)
            loss_ris = self.loss.receiver_imitation_loss(result_s)
            loss_ri = torch.stack(list(loss_ris.values()), dim=-1).mean(dim=-1)
            total_loss = loss_r + loss_s + loss_ri
            agent_metrics[name_s] = {
                f"acc_comp.mean": fmean(acc_comps),
                f"acc_part.mean": fmean(acc_parts),
                f"acc_comp.max": max(acc_comps),
                f"acc_part.max": max(acc_parts),
                f"acc_comp.min": min(acc_comps),
                f"acc_part.min": min(acc_parts),
                f"receiver_loss": loss_r.mean().item(),
                f"sender_loss": loss_s.mean().item(),
                f"receiver_imitation_loss": loss_ri.mean().item(),
                f"total_loss": total_loss.mean().item(),
                f"entropy": result_s.message_entropy.mean().item(),
                f"length": result_s.message_length.float().mean().item(),
                f"unique": result_s.message.unique(dim=0).shape[0]
                / result_s.message.shape[0],
            }

        metrics = {
            f"{name}.{k}": v for name, m in agent_metrics.items() for k, v in m.items()
        }
        keys = next(iter(agent_metrics.values())).keys()
        metrics |= {
            f"{k}.mean": fmean([m[k] for m in agent_metrics.values()]) for k in keys
        }

        for logger in self.loggers:
            logger.log(metrics, step)


class AccuracyHeatmapLogger(Loggable[RecoNetworkGameResult]):
    def __init__(
        self,
        acc_comp_heatmap_logger: HeatmapLogger,
        acc_part_heatmap_logger: HeatmapLogger,
    ) -> None:
        self.acc_comp_heatmap_logger = acc_comp_heatmap_logger
        self.acc_part_heatmap_logger = acc_part_heatmap_logger

    def log(self, result: RecoNetworkGameResult, step: int | None = None) -> None:
        names = list(result.agents)
        matrix_comp: List[List[float]] = []
        matrix_part: List[List[float]] = []
        for name_s in names:
            comps, parts = [], []
            for name_r in names:
                acc_comp, acc_part = acc_comp_part(
                    result.sub_results[name_s].outputs[name_r], result.input
                )
                comps.append(acc_comp)
                parts.append(acc_part)

            matrix_comp.append(comps)
            matrix_part.append(parts)

        for matrix, logger in [
            (matrix_comp, self.acc_comp_heatmap_logger),
            (matrix_part, self.acc_part_heatmap_logger),
        ]:
            df = pd.DataFrame(matrix, columns=names, index=names)
            logger.log(df, step)


def acc_comp_part(
    input: TensorType[..., int],
    target: TensorType[..., int],
) -> Tuple[float, float]:
    mark = target == input
    acc_comp = mark.all(dim=-1).float().mean().item()
    acc_part = mark.float().mean(dim=0).mean().item()
    return acc_comp, acc_part
