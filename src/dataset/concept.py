import itertools

import torch as th


def concept_dataset(
    n_attributes: int,
    n_values: int,
    device: th.device | str = "cpu",
) -> th.Tensor:
    dataset = (
        th.Tensor(list(itertools.product(th.arange(n_values), repeat=n_attributes)))
        .long()
        .to(device)
    )
    return dataset


def onehot_concept_dataset(n_attributes: int, n_values: int) -> th.Tensor:
    dataset = concept_dataset(n_attributes, n_values)
    dataset = th.nn.functional.one_hot(dataset, n_values).float()
    dataset = dataset.view(-1, n_attributes * n_values)
    return dataset
