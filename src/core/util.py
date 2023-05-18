import os
from typing import Callable

import editdistance
import numpy as np
import torch as th
from networkx import DiGraph
from numba import jit
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


class ModelSaver(Callback):
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
                    agent.model,
                    f"{save_dir}/{self.count}.pth",
                )

        self.count += 1


class ModelInitializer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: DiGraph,
    ):
        super().__init__()
        self.agents = agents
        self.network = network

        self._nodes = list(self.network.nodes)

    def on_begin(self):
        for agent_name in self._nodes:
            agent = self.agents[agent_name]
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
    :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
    :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
    If no <eos> is found, the full length is returned (i.e. messages.size(1)).

    >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)
    zero_mask = messages == 0
    # a bit involved logic, but it seems to be faster for large batches than slicing batch dimension and
    # querying torch.nonzero()
    # zero_mask contains ones on positions where 0 occur in the outputs, and 1 otherwise
    # zero_mask.cumsum(dim=1) would contain non-zeros on all positions after 0 occurred
    # zero_mask.cumsum(dim=1) > 0 would contain ones on all positions after 0 occurred
    # (zero_mask.cumsum(dim=1) > 0).sum(dim=1) equates to the number of steps  happened after 0 occured (including it)
    # max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before 0 took place

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths


def message_similarity(
    message1: th.Tensor,
    message2: th.Tensor,
    length1: th.Tensor | None = None,
    length2: th.Tensor | None = None,
):
    if length1 is None:
        length1 = find_length(message1)
    if length2 is None:
        length2 = find_length(message2)

    message1 = message1.cpu().numpy()
    message2 = message2.cpu().numpy()

    tensor1_trimmed = [np.trim_zeros(seq, trim="b") for seq in message1]
    tensor2_trimmed = [np.trim_zeros(seq, trim="b") for seq in message2]

    edit_distances = [
        editdistance.eval(seq1, seq2)
        for seq1, seq2 in zip(tensor1_trimmed, tensor2_trimmed)
    ]

    return 1 - th.tensor(edit_distances, dtype=th.int) / th.max(length1, length2)


# def concept_distance(concept1: np.ndarray, concept2: np.ndarray):
#     concept1 = th.tensor(concept1)
#     concept2 = th.tensor(concept2)
#     return (concept1 != concept2).float().mean(dim=-1)


@jit(nopython=True)
def pdist(
    X: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    n = X.shape[0]
    out_size = (n * (n - 1)) // 2
    dm = list(range(out_size))
    k = 0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            dm[k] = metric(X[i], X[j])
            k += 1
    return dm


@jit(nopython=True)
def concept_distance(concept1: np.ndarray, concept2: np.ndarray):
    return np.mean(concept1 != concept2)


@jit(nopython=True)
def edit_distance(s1: np.ndarray, s2: np.ndarray):
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


def topsim_numbapdist_numbaedit(
    concept: np.ndarray, language: np.ndarray, corr: str = "spearman"
):
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
