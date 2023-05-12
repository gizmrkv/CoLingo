import os

import editdistance
import numpy as np
import torch as th

from .agent import Agent
from .callback import Callback


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
                    agent.model,
                    f"{save_dir}/{self.count}.pth",
                )

        self.count += 1


def fix_seed(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def concept_distance(concept1: np.ndarray, concept2: np.ndarray):
    concept1 = th.tensor(concept1)
    concept2 = th.tensor(concept2)
    return (concept1 != concept2).float().mean(dim=-1)
