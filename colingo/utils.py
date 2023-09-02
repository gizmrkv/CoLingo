import random
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType


def fix_seed(seed: int) -> None:
    """
    Fix random seed for reproducibility of random operations.

    Args:
        seed (int): Seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(m: nn.Module) -> None:
    """
    Initialize the weights of a neural network module.

    Args:
        m (nn.Module): The neural network module to initialize.
    """

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
        if isinstance(m.weight_ih_l0, torch.Tensor):
            nn.init.kaiming_uniform_(m.weight_ih_l0)
        if isinstance(m.weight_hh_l0, torch.Tensor):
            nn.init.kaiming_uniform_(m.weight_hh_l0)
        if isinstance(m.bias_ih_l0, torch.Tensor):
            nn.init.zeros_(m.bias_ih_l0)
        if isinstance(m.bias_hh_l0, torch.Tensor):
            nn.init.zeros_(m.bias_hh_l0)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def random_split(dataset: TensorType, proportions: Iterable[float]) -> List[TensorType]:
    """
    Randomly split a dataset into multiple subsets according to given proportions.

    Args:
        dataset (TensorType): The dataset to split.
        proportions (Iterable[float]): Proportions for splitting the dataset.

    Returns:
        List[TensorType]: List of subsets of the dataset.
    """
    indices = np.random.permutation(len(dataset))

    proportions_sum = sum(proportions)
    split_sizes = [int(r / proportions_sum * len(dataset)) for r in proportions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[torch.tensor(idx)] for idx in split_indices]

    return split_dataset
