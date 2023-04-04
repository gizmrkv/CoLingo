import toml

from src.model import build_model
from src.network import build_network
from src.dataset import build_dataset
from src.dataloader import build_dataloader
from src.game import build_game
from src.agent import Agent
from src.trainer import Trainer


def main(config: dict):
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

    games = {
        game_name: build_game(
            game_info["type"],
            dataloaders[game_info["dataloader"]],
            agents,
            networks[game_info["network"]],
            game_info["args"],
        )
        for game_name, game_info in config["games"].items()
    }

    trainer = Trainer(games)
    trainer.train(10000, 1000)

    message = agents["smith"](datasets["onehots1"], "object", is_training=False)
    answer = agents["brown"](message, "message", is_training=False)
    object = datasets["onehots1"].argmax(dim=-1)
    message = message.argmax(dim=-1)
    answer = answer.argmax(dim=-1)
    print(f"{object} -> {message} -> {answer}")


if __name__ == "__main__":
    with open("config/sample.toml", "r") as f:
        config = toml.load(f)

    main(config)
