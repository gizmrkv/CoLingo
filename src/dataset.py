import torch as th
import itertools


def build_onthots_dataset(n_attributes: int, n_values: int) -> th.Tensor:
    dataset = th.Tensor(
        list(itertools.product(th.arange(n_values), repeat=n_attributes))
    ).long()
    dataset = th.nn.functional.one_hot(dataset, n_values).float()
    dataset = dataset.view(-1, n_attributes * n_values)
    return dataset


def build_normal_dataset(n_dim: int, mean: float, std: float) -> th.Tensor:
    dataset = th.normal(mean, std, (n_dim,))
    return dataset


def build_dataset(dataset_type: str, dataset_args: dict) -> th.Tensor:
    datasets_dict = {"onehots": build_onthots_dataset, "normal": build_normal_dataset}
    return datasets_dict[dataset_type](**dataset_args)
