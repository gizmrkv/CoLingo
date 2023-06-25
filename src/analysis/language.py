from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch as th
from numba import njit
from scipy.stats import kendalltau, pearsonr, spearmanr

from ..core.callback import Callback
from ..logger import Logger
from .concept import concept_distance
from .util import edit_distance, pdist


def language_uniques(
    language: th.Tensor,
) -> int:
    language = language.sort(dim=1)[0]
    language = th.unique(language, dim=0)
    n_uniques = language.shape[0]
    return n_uniques


def concept_topographic_similarity(
    concept: np.ndarray, language: np.ndarray, corr: str = "spearman"
) -> float:
    concept_pdist = pdist(concept, concept_distance)
    language_pdist = pdist(language, edit_distance)

    if corr == "spearman":
        corr = spearmanr
    elif corr == "kendall":
        corr = kendalltau
    elif corr == "pearson":
        corr = pearsonr
    else:
        raise ValueError("corr must be spearman or kendall")
    return corr(concept_pdist, language_pdist).correlation


@njit
def language_similarity(
    language1: np.ndarray,
    language2: np.ndarray,
    length1: np.ndarray | None = None,
    length2: np.ndarray | None = None,
    distance: str = "edit_distance",
) -> float:
    if length1 is None:
        length1 = np.argmin(language1, axis=1) + 1
    if length2 is None:
        length2 = np.argmin(language2, axis=1) + 1

    if distance == "edit_distance":
        distance = edit_distance
    else:
        raise ValueError("distance must be edit_distance")

    distances = np.arange(language1.shape[0], dtype=np.int64)
    for i in range(language1.shape[0]):
        distances[i] = edit_distance(
            language1[i, : length1[i]], language2[i, : length2[i]]
        )

    return 1 - (distances / np.maximum(length1, length2)).mean()


@dataclass
class LanguageResult:
    input: th.Tensor
    languages: dict[str, Any]


class LanguageEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, th.nn.Module],
        input: th.Tensor,
        metric: Callable[[LanguageResult], dict],
        logger: Logger | Iterable[Logger],
        name: str,
        run_on_begin: bool = False,
        run_on_end: bool = True,
        input_command: str = "input",
        send_command: str = "send",
    ):
        super().__init__()
        self.agents = agents
        self.input = input
        self.metric = metric
        self.loggers = [logger] if isinstance(logger, Logger) else logger
        self.name = name
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end
        self.input_command = input_command
        self.send_command = send_command

    def on_begin(self):
        if self.run_on_begin:
            self.evaluate()
            for logger in self.loggers:
                logger.flush()

    def on_end(self):
        if self.run_on_end:
            self.evaluate()
            for logger in self.loggers:
                logger.flush()

    def on_update(self, iteration: int):
        self.evaluate()

    def evaluate(self):
        result = LanguageResult(
            input=self.input,
            languages={},
        )
        for agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.eval()
            with th.no_grad():
                latent = agent(input=self.input, command=self.input_command)
                message = agent(latent=latent, command=self.send_command)

            result.languages[agent_name] = message

        metric = self.metric(result)

        for logger in self.loggers:
            logger.log({self.name: metric})
