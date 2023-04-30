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


dataset_types = {"onehots": build_onehots_dataset, "normal": build_normal_dataset}
model_types = {"single_word": SingleWordModel, "sequence": SequenceModel}
losse_types = {"reinforce": ReinforceLoss}
baseline_types = {"mean": MeanBaseline}
task_types = {"lewis": LewisGame}
network_types = {"custom": CustomNetwork}
dataloader_types = {"default": th.utils.data.DataLoader}
optimizer_types = {
    "adam": th.optim.Adam,
    "sgd": th.optim.SGD,
    "adagrad": th.optim.Adagrad,
    "adadelta": th.optim.Adadelta,
    "rmsprop": th.optim.RMSprop,
    "sparseadam": th.optim.SparseAdam,
    "adamax": th.optim.Adamax,
    "asgd": th.optim.ASGD,
    "lbfgs": th.optim.LBFGS,
    "rprop": th.optim.Rprop,
}


def build_datasets(datasets_config: dict[str, dict]):
    datasets = {}
    for name, params in datasets_config.copy().items():
        dataset_type = params["type"]
        dataset_params = {k: v for k, v in params.items() if k != "type"}
        datasets[name] = dataset_types[dataset_type](**dataset_params)
    return datasets


def build_agents(agents_config: dict[str, dict]):
    agents = {}
    for name, params in agents_config.items():
        agent_params = {k: v for k, v in params.items() if k != "type"}

        # build model
        if "model" in agent_params.keys():
            model_type = agent_params["model"]["type"]
            model_params = {
                k: v for k, v in agent_params["model"].items() if k != "type"
            }
            agent_params["model"] = model_types[model_type](**model_params)

        # setting for each tasks
        if "tasks" in agent_params.keys():
            for task_name, task_params in agent_params["tasks"].items():
                agent_params["tasks"][task_name] = task_params.copy()

                # build optimizer
                if "optimizer" in task_params.keys():
                    optimizer_type = task_params["optimizer"]["type"]
                    optimizer_params = {
                        k: v for k, v in task_params["optimizer"].items() if k != "type"
                    }
                    agent_params["tasks"][task_name]["optimizer"] = optimizer_types[
                        optimizer_type
                    ]
                    agent_params["tasks"][task_name][
                        "optimizer_params"
                    ] = optimizer_params

                # build loss
                if "loss" in task_params.keys():
                    loss_type = task_params["loss"]["type"]
                    loss_params = {
                        k: v for k, v in task_params["loss"].items() if k != "type"
                    }
                    agent_params["tasks"][task_name]["loss"] = losse_types[loss_type](
                        **loss_params
                    )

        agents[name] = Agent(name=name, **agent_params)

    return agents


def build_tasks(
    tasks_config: dict[str, dict], agents: dict[str, Agent], datasets: dict
):
    tasks = {}
    for name, params in tasks_config.copy().items():
        task_type = params["type"]
        task_params = {k: v for k, v in params.items() if k != "type"}

        # build network
        if "network" in task_params.keys():
            network_type = task_params["network"]["type"]
            network_params = {
                k: v for k, v in task_params["network"].items() if k != "type"
            }
            task_params["network"] = network_types[network_type](
                agents=agents, **network_params
            )

        # build dataloader
        if "dataloader" in task_params.keys():
            dataloader_type = task_params["dataloader"]["type"]
            dataloader_params = {
                k: v for k, v in task_params["dataloader"].items() if k != "type"
            }
            dataloader_params["dataset"] = datasets[dataloader_params["dataset"]]
            task_params["dataloader"] = dataloader_types[dataloader_type](
                **dataloader_params
            )

        # build task
        tasks[name] = task_types[task_type](name=name, **task_params)

    return tasks


def check_config(datasets, agents, tasks):
    # TODO: implemnet
    pass


def main(config: dict):
    date = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
    exp_id = random.randint(0, 100000)
    exp_dir = f"exp/{config['exp_name']} {date} {exp_id}"

    fix_seed(config["seed"])

    datasets = build_datasets(config["datasets"])
    agents = build_agents(config["agents"])
    tasks = build_tasks(config["tasks"], agents, datasets)

    check_config(datasets, agents, tasks)

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
