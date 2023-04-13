import toml
from pprint import pprint

import torch as th
from torch.utils.data import DataLoader

from src.dataset import build_onthots_dataset, build_normal_dataset
from src.model import SingleWordModel, SequenceModel
from src.baseline import MeanBaseline
from src.agent import Agent
from src.network import Network, CustomNetwork
from src.task import Task, LewisGame


class ValidationGame(Task):
    def __init__(
        self,
        network: Network,
        dataloader: DataLoader,
        interval: float,
    ):
        super().__init__()

        self.network = network
        self.dataset = dataloader.dataset
        self.interval = interval

        self.count = 0

    def run(self):
        if self.count % self.interval == 0:
            sender = self.network.agents["S01"]
            receiver1 = self.network.agents["R01"]

            for ag in [sender, receiver1]:
                ag.eval()

            message = sender(self.dataset, "object")
            answer1 = receiver1(message, "message")

            # object = self.dataset.argmax(dim=-1)
            # message = message.argmax(dim=-1)
            answer1 = answer1.argmax(dim=-1)

            print(f"{message} -> {answer1}")

        self.count += 1


def fix_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_config(config: dict) -> dict:
    types = {
        "datasets": {
            "onehots": build_onthots_dataset,
            "normal": build_normal_dataset,
        },
        "dataloaders": {"default": th.utils.data.DataLoader},
        "models": {
            "single_word": SingleWordModel,
            "sequence": SequenceModel,
        },
        "baselines": {"mean": MeanBaseline},
        "agents": {"agent": Agent},
        "networks": {"custom": CustomNetwork},
        "tasks": {"lewis": LewisGame},
    }
    entries = [
        "datasets",
        "dataloaders",
        "models",
        "baselines",
        "agents",
        "networks",
        "tasks",
    ]

    for entry in entries:
        if entry in config.keys():
            for name, type_args in config[entry].items():
                type = type_args["type"]
                args = type_args["args"]
                for key, value in args.items():
                    if key + "s" in types.keys() and value in config[key + "s"].keys():
                        args[key] = config[key + "s"][value]

                if entry == "networks":
                    instance = types[entry][type](agents=config["agents"], **args)
                else:
                    instance = types[entry][type](**args)
                config[entry][name] = instance


def main(config: dict):
    fix_seed(config["training"]["seed"])

    update_config(config)

    validation = ValidationGame(
        network=config["networks"]["custom_net1"],
        dataloader=config["dataloaders"]["onehots_loader1"],
        interval=100,
    )
    config["tasks"]["validation"] = validation

    for epoch in range(config["training"]["n_epochs"]):
        for name, ts in config["tasks"].items():
            ts.run()


if __name__ == "__main__":
    with open("config/sample.toml", "r") as f:
        config = toml.load(f)

    main(config)
