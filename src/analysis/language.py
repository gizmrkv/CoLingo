from typing import Callable, Iterable

import numpy as np
import torch as th
from numba import njit
from scipy.stats import kendalltau, pearsonr, spearmanr

from ..agent import Agent
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


class LanguageEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        input: th.Tensor,
        metric: Callable[
            [np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]], dict
        ],
        logger: Logger | Iterable[Logger],
        name: str,
    ):
        super().__init__()
        self.agents = agents
        self.input = input
        self.metric = metric
        self.loggers = [logger] if isinstance(logger, Logger) else logger
        self.name = name

    def on_end(self):
        languages = {}
        lengths = {}
        for agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.eval()
            with th.no_grad():
                hidden = agent.input(input=self.input)
                language, log_prob, entropy, length = agent.message(
                    hidden, game_name=self.name
                )

            languages[agent_name] = language.cpu().numpy()
            lengths[agent_name] = length.cpu().numpy()

        metric = self.metric(self.input.cpu().numpy(), languages, lengths)

        for logger in self.loggers:
            logger.log({self.name: metric}, flush=True)
