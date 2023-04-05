import torch as th
from torch.utils.data import DataLoader


def build_dataloader(
    dataloader_type: str, dataset: th.Tensor, dataloader_args: dict
) -> DataLoader:
    dataloaders_dict = {"default": DataLoader}
    return dataloaders_dict[dataloader_type](dataset, **dataloader_args)
