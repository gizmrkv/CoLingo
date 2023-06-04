import numpy as np
import torch as th


def random_split(dataset: th.Tensor, propotions: list[float]) -> list[th.Tensor]:
    indices = np.random.permutation(len(dataset))

    propotions_sum = sum(propotions)
    split_sizes = [int(r / propotions_sum * len(dataset)) for r in propotions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[th.tensor(idx)] for idx in split_indices]

    return split_dataset
