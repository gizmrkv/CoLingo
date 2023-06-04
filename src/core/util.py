import os
from typing import Callable, Iterable

import numpy as np
import torch as th
from numba import njit
from scipy.stats import kendalltau, pearsonr, spearmanr

from .agent import Agent
from .callback import Callback


def fix_seed(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AgentSaver(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        interval: float,
        path: str,
    ):
        super().__init__()

        self.agents = agents
        self.interval = interval
        self.path = path

        self.count = 0

    def on_update(self):
        if self.count % self.interval == 0:
            for agent_name, agent in self.agents.items():
                save_dir = f"{self.path}/{agent_name}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                th.save(
                    agent,
                    f"{save_dir}/{self.count}.pth",
                )

        self.count += 1


class AgentInitializer(Callback):
    def __init__(
        self,
        agents: Iterable[Agent],
    ):
        super().__init__()
        self.agents = agents

    def on_begin(self):
        for agent in self.agents:
            agent.apply(init_weights)


def init_weights(m):
    if isinstance(m, (th.nn.Linear, th.nn.Conv2d)):
        th.nn.init.kaiming_uniform_(m.weight)
        th.nn.init.zeros_(m.bias)
    elif isinstance(m, (th.nn.RNN, th.nn.LSTM, th.nn.GRU)):
        th.nn.init.kaiming_uniform_(m.weight_ih_l0)
        th.nn.init.kaiming_uniform_(m.weight_hh_l0)
        th.nn.init.zeros_(m.bias_ih_l0)
        th.nn.init.zeros_(m.bias_hh_l0)
    elif isinstance(m, th.nn.Embedding):
        th.nn.init.kaiming_uniform_(m.weight)
    elif isinstance(
        m, (th.nn.LayerNorm, th.nn.BatchNorm1d, th.nn.BatchNorm2d, th.nn.BatchNorm3d)
    ):
        th.nn.init.constant_(m.weight, 1)
        th.nn.init.constant_(m.bias, 0)


def find_length(messages: th.Tensor) -> th.Tensor:
    """
    must has 0 at the end of each sequence
    """
    return messages.argmin(dim=1) + 1


@njit
def pdist(X: np.ndarray, dist: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    n = X.shape[0]
    size = (n * (n - 1)) // 2
    distances = list(range(size))
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            distances[k] = dist(X[i], X[j])
            k += 1
    return distances


@njit
def concept_distance(concept1: np.ndarray, concept2: np.ndarray):
    return np.mean(concept1 != concept2)


@njit
def edit_distance(s1: np.ndarray, s2: np.ndarray):
    """
    This is slower than the editdistance.eval, but it is compatible with numba.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = np.arange(len(s2) + 1, dtype=np.int64)
    for i, c1 in enumerate(s1):
        current_row = np.zeros(len(s2) + 1, dtype=np.int64)
        current_row[0] = i + 1
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def topographic_similarity(
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

    return 1 - distances / np.maximum(length1, length2)
