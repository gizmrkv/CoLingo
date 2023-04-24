import toml
import yaml
from pprint import pprint

import torch as th
from torch.utils.data import DataLoader

from src.util import fix_seed, find_length

from src.dataset import build_onehots_dataset, build_normal_dataset
from src.model import SingleWordModel, SequenceModel
from src.loss import ReinforceLoss
from src.baseline import MeanBaseline
from src.agent import Agent
from src.network import Network, CustomNetwork
from src.task import Task, LewisGame, ModelSaver


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
            sender = self.network.agents["agent1"]
            receiver1 = self.network.agents["agent2"]

            for ag in [sender, receiver1]:
                ag.eval()

            message, _ = sender(self.dataset, "object")
            answer1, _ = receiver1(message, "message")

            # object = self.dataset.argmax(dim=-1)
            # message = message.argmax(dim=-1)
            answer1 = answer1.argmax(dim=-1)

            # print(f"{message} -> {answer1}")

            acc = (answer1 == self.dataset.argmax(dim=-1)).float().mean()
            print(f"Epoch: {self.count}")
            print(f"Accuracy: {acc:.3f}")
            for obj, msg, ans in zip(self.dataset, message, answer1):
                print(f"{obj.argmax(dim=-1)} -> {msg.tolist()[:-1]} -> {ans}")

        self.count += 1


def build_instance(types: dict[str, dict], type_params: dict[str, dict]):
    if "params" not in type_params.keys() or type_params["params"] is None:
        return types[type_params["type"]]()
    else:
        return types[type_params["type"]](**type_params["params"])


datasets_type = {"onehots": build_onehots_dataset, "normal": build_normal_dataset}
models_type = {"single_word": SingleWordModel, "sequence": SequenceModel}
losses_type = {"reinforce": ReinforceLoss}
baselines_type = {"mean": MeanBaseline}
tasks_type = {"lewis": LewisGame}
networks_type = {"custom": CustomNetwork}
dataloaders_type = {"default": th.utils.data.DataLoader}


def build_datasets(datasets_config: dict[str, dict]):
    datasets = {}
    for name, type_params in datasets_config.copy().items():
        datasets[name] = build_instance(datasets_type, type_params)
    return datasets


def build_agents(agents_config: dict[str, dict]):
    agents = {}
    for name, params in agents_config.copy().items():
        if "model" in params.keys():
            params["model"] = build_instance(models_type, params["model"])

        if "loss" in params.keys():
            loss_params = params["loss"]["params"]
            if "baseline" in loss_params.keys():
                loss_params["baseline"] = build_instance(
                    baselines_type, loss_params["baseline"]
                )

            params["loss"] = build_instance(losses_type, params["loss"])

        params["name"] = name
        agents[name] = Agent(**params)

    return agents


def build_tasks(
    tasks_config: dict[str, dict], agents: dict[str, Agent], datasets: dict
):
    tasks = {}
    for name, type_params in tasks_config.copy().items():
        if "network" in type_params["params"].keys():
            network_params = type_params["params"]["network"]
            network_params["params"]["agents"] = agents
            type_params["params"]["network"] = build_instance(
                networks_type, network_params
            )
        if "dataloader" in type_params["params"].keys():
            dataloader_params = type_params["params"]["dataloader"]
            dataloader_params["params"]["dataset"] = datasets[
                dataloader_params["params"]["dataset"]
            ]
            type_params["params"]["dataloader"] = build_instance(
                dataloaders_type, dataloader_params
            )
        tasks[name] = build_instance(tasks_type, type_params)

    return tasks


def main(config: dict):
    fix_seed(config["seed"])

    datasets = build_datasets(config["datasets"])
    agents = build_agents(config["agents"])
    tasks = build_tasks(config["tasks"], agents, datasets)

    validation = ValidationGame(
        network=tasks["task1"].network,
        dataloader=tasks["task1"].dataloader,
        interval=10,
    )
    tasks["validation"] = validation

    tasks["model_saver"] = ModelSaver(agents, 1000, "hoge/models")

    for epoch in range(config["n_epochs"]):
        for name, task in tasks.items():
            task.run()


if __name__ == "__main__":
    with open("config/sample.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(config)
