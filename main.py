import toml

from torch.utils.data import DataLoader

from src.model import build_model
from src.network import build_network, Network
from src.dataset import build_dataset
from src.dataloader import build_dataloader
from src.game import build_game, Game
from src.agent import Agent
from src.trainer import Trainer


class ValidationGame(Game):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: Network,
        dataloader: DataLoader,
        play_rate: float,
    ):
        super().__init__(agents, network, dataloader, play_rate)
        self.dataset = dataloader.dataset
        self.count = 0

    def play(self):
        sender = self.agents["S01"]
        receiver1 = self.agents["R01"]

        for agent in [sender, receiver1]:
            agent.eval()

        message = sender(self.dataset, "object")
        answer1 = receiver1(message, "message")

        # object = self.dataset.argmax(dim=-1)
        # message = message.argmax(dim=-1)
        answer1 = answer1.argmax(dim=-1)

        print(f"{message} -> {answer1}")


def fix_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config: dict):
    fix_seed(config["training"]["seed"])

    datasets = {
        dataset_name: build_dataset(dataset_info["type"], dataset_info["args"])
        for dataset_name, dataset_info in config["datasets"].items()
    }

    dataloaders = {
        dataloader_name: build_dataloader(
            dataloader_info["type"],
            datasets[dataloader_info["dataset"]],
            dataloader_info["args"],
        )
        for dataloader_name, dataloader_info in config["dataloaders"].items()
    }

    networks = {
        network_name: build_network(network_info["type"], network_info["args"])
        for network_name, network_info in config["networks"].items()
    }

    models = {
        model_name: build_model(model_info["type"], model_info["args"])
        for model_name, model_info in config["models"].items()
    }

    agents = {
        agent_name: Agent(models[agent_info["model"]], **agent_info["args"])
        for agent_name, agent_info in config["agents"].items()
    }

    for agent in agents.values():
        agent.train()

    games = {
        game_name: build_game(
            game_info["type"],
            agents,
            networks[game_info["network"]],
            dataloaders[game_info["dataloader"]],
            game_info["play_rate"],
            game_info["args"],
        )
        for game_name, game_info in config["games"].items()
    }

    validation = ValidationGame(
        agents,
        network=networks["custom_net1"],
        dataloader=dataloaders["onehots_loader1"],
        play_rate=0.001,
    )
    games["validation"] = validation

    trainer = Trainer(games)
    trainer.train(config["training"]["n_epochs"])


if __name__ == "__main__":
    with open("config/sample.toml", "r") as f:
        config = toml.load(f)

    main(config)
