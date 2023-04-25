import datetime
import random
from pprint import pprint

import toml
import torch as th
import yaml
from torch.utils.data import DataLoader

from src.agent import Agent
from src.baseline import MeanBaseline
from src.dataset import build_normal_dataset, build_onehots_dataset
from src.loss import ReinforceLoss
from src.model import SequenceModel, SingleWordModel
from src.network import CustomNetwork, Network
from src.task import AgentSaver, LewisGame, Task
from src.util import find_length, fix_seed


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
            answer, _ = receiver1(message, "message")

            n_attributes = 2
            n_data = self.dataset.shape[0]
            dataset = (
                self.dataset.view(n_data * n_attributes, -1)
                .argmax(dim=-1)
                .reshape(-1, n_attributes)
            )
            answer = (
                answer.view(n_data * n_attributes, -1)
                .argmax(dim=-1)
                .reshape(-1, n_attributes)
            )

            acc = (answer == dataset).float().mean()
            print(f"Epoch: {self.count}")
            print(f"Accuracy: {acc:.3f}")
            for obj, msg, ans in zip(dataset, message, answer):
                print(
                    f"{tuple(obj.tolist())} -> {msg.tolist()[:-1]} -> {tuple(ans.tolist())}"
                )

        self.count += 1


def build_instance(types: dict[str, dict], spec: dict[str, dict]):
    if "params" not in spec.keys() or spec["params"] is None:
        return types[spec["type"]]()
    else:
        return types[spec["type"]](**spec["params"])


agents_type = {"create": Agent, "load": th.load}
datasets_type = {"onehots": build_onehots_dataset, "normal": build_normal_dataset}
models_type = {"single_word": SingleWordModel, "sequence": SequenceModel}
losses_type = {"reinforce": ReinforceLoss}
baselines_type = {"mean": MeanBaseline}
tasks_type = {"lewis": LewisGame}
networks_type = {"custom": CustomNetwork}
dataloaders_type = {"default": th.utils.data.DataLoader}


def build_datasets(datasets_config: dict[str, dict]):
    datasets = {}
    for name, spec in datasets_config.copy().items():
        datasets[name] = build_instance(datasets_type, spec)
    return datasets


def build_agents(agents_config: dict[str, dict]):
    agents = {}
    for name, agent_spec in agents_config.copy().items():
        if "params" in agent_spec.keys():
            if "model" in agent_spec["params"].keys():
                agent_spec["params"]["model"] = build_instance(
                    models_type, agent_spec["params"]["model"]
                )

            if "loss" in agent_spec["params"].keys():
                loss_spec = agent_spec["params"]["loss"]
                if "params" in loss_spec.keys():
                    if "baseline" in loss_spec["params"].keys():
                        loss_spec["params"]["baseline"] = build_instance(
                            baselines_type, loss_spec["params"]["baseline"]
                        )

                agent_spec["params"]["loss"] = build_instance(losses_type, loss_spec)

        agents[name] = build_instance(agents_type, agent_spec)
        agents[name].name = name

    return agents


def build_tasks(
    tasks_config: dict[str, dict], agents: dict[str, Agent], datasets: dict
):
    tasks = {}
    for name, task_spec in tasks_config.copy().items():
        if "params" in task_spec.keys():
            if "network" in task_spec["params"].keys():
                network_params = task_spec["params"]["network"]
                network_params["params"]["agents"] = agents
                task_spec["params"]["network"] = build_instance(
                    networks_type, network_params
                )
            if "dataloader" in task_spec["params"].keys():
                dataloader_params = task_spec["params"]["dataloader"]
                dataloader_params["params"]["dataset"] = datasets[
                    dataloader_params["params"]["dataset"]
                ]
                task_spec["params"]["dataloader"] = build_instance(
                    dataloaders_type, dataloader_params
                )
        tasks[name] = build_instance(tasks_type, task_spec)

    return tasks


def main(config: dict):
    date = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
    exp_id = random.randint(0, 100000)
    exp_dir = f"exp/{config['exp_name']} {date} {exp_id}"

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

    tasks["model_saver"] = AgentSaver(agents, 1000, f"{exp_dir}/models")

    for epoch in range(config["n_epochs"]):
        for name, task in tasks.items():
            task.run()


if __name__ == "__main__":
    with open("config/sample.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(config)
