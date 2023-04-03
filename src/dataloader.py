import torch as th


def build_dataloader(dataloader_type: str, dataset: th.Tensor, dataloader_args: dict):
    dataloaders_dict = {"default": th.utils.data.DataLoader}
    return dataloaders_dict[dataloader_type](dataset, **dataloader_args)
