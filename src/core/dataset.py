import itertools
from typing import Callable

import numpy as np
import torch as th
from torch.utils.data import TensorDataset


def create_concept_dataset(
    n_attributes: int,
    n_values: int,
    transform: Callable[[th.Tensor], th.Tensor] | None = None,
    device: th.device | str = "cpu",
) -> th.Tensor:
    dataset = (
        th.Tensor(list(itertools.product(th.arange(n_values), repeat=n_attributes)))
        .long()
        .to(device)
    )

    if transform is None:
        target = dataset
    else:
        target = transform(dataset)
    return TensorDataset(dataset, target)


def create_onehot_concept_dataset(n_attributes: int, n_values: int) -> th.Tensor:
    dataset = create_concept_dataset(n_attributes, n_values)
    dataset = th.nn.functional.one_hot(dataset, n_values).float()
    dataset = dataset.view(-1, n_attributes * n_values)
    return dataset


def create_normal_dataset(n_dim: int, mean: float, std: float) -> th.Tensor:
    dataset = th.normal(mean, std, (n_dim,))
    return dataset


def random_split(dataset: th.Tensor, propotions: list[float]) -> list[th.Tensor]:
    indices = np.random.permutation(len(dataset))

    propotions_sum = sum(propotions)
    split_sizes = [int(r / propotions_sum * len(dataset)) for r in propotions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[th.tensor(idx)] for idx in split_indices]

    return split_dataset
