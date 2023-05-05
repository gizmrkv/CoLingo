import torch as th
import itertools


def build_concept_dataset(n_attributes: int, n_values: int) -> th.Tensor:
    dataset = th.Tensor(
        list(itertools.product(th.arange(n_values), repeat=n_attributes))
    ).long()
    return dataset


def build_onehot_concept_dataset(n_attributes: int, n_values: int) -> th.Tensor:
    dataset = build_concept_dataset(n_attributes, n_values)
    dataset = th.nn.functional.one_hot(dataset, n_values).float()
    dataset = dataset.view(-1, n_attributes * n_values)
    return dataset


def build_normal_dataset(n_dim: int, mean: float, std: float) -> th.Tensor:
    dataset = th.normal(mean, std, (n_dim,))
    return dataset
