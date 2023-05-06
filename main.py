import datetime
import random
from pprint import pprint

import toml
import torch as th
import yaml
from torch.utils.data import DataLoader

from src.agent import Agent
from src.baseline import MeanBaseline, BatchMeanBaseline
from src.dataset import (
    build_normal_dataset,
    build_onehot_concept_dataset,
    build_concept_dataset,
    random_split,
)
from src.loss import ReinforceLoss, ConceptLoss, OnehotConceptLoss
from src.model import (
    OnehotConceptSequntialMessageModel,
    OnehotConceptSymbolMessageModel,
    EmbeddingConceptSequentialMessageModel,
)
from src.network import CustomNetwork, Network
from src.task import AgentSaver, SignalingTrainer, Task
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
            # dataset = (
            #     self.dataset.view(n_data * n_attributes, -1)
            #     .argmax(dim=-1)
            #     .reshape(-1, n_attributes)
            # )
            dataset = self.dataset
            answer = (
                answer.view(n_data * n_attributes, -1)
                .argmax(dim=-1)
                .reshape(-1, n_attributes)
            )

            acc = (answer == dataset).float().mean()
            for obj, msg, ans in zip(dataset, message, answer):
                print(
                    f"{tuple(obj.tolist())} -> {msg.tolist()[:-1]} -> {tuple(ans.tolist())}"
                )
            print(f"Epoch: {self.count:4}, Accuracy: {acc:.3f}")

        self.count += 1


dataset_types = {
    "concept": build_concept_dataset,
    "onehot_concept": build_onehot_concept_dataset,
    "normal": build_normal_dataset,
    "random_split": random_split,
}
model_types = {
    "ocsym": OnehotConceptSymbolMessageModel,
    "ocsem": OnehotConceptSequntialMessageModel,
    "ecsem": EmbeddingConceptSequentialMessageModel,
}
loss_types = {
    "reinforce": ReinforceLoss,
    "concept": ConceptLoss,
    "onehot_concept": OnehotConceptLoss,
}
baseline_types = {"mean": MeanBaseline, "batch_mean": BatchMeanBaseline}
task_types = {"signaling": SignalingTrainer}
network_types = {"custom": CustomNetwork}
dataloader_types = {"builtin": th.utils.data.DataLoader}
optimizer_types = {
    "adadelta": th.optim.Adadelta,
    "adagrad": th.optim.Adagrad,
    "adam": th.optim.Adam,
    "adamax": th.optim.Adamax,
    "asgd": th.optim.ASGD,
    "lbfgs": th.optim.LBFGS,
    "rmsprop": th.optim.RMSprop,
    "rprop": th.optim.Rprop,
    "sgd": th.optim.SGD,
    "sparseadam": th.optim.SparseAdam,
    "adamw": th.optim.AdamW,
    "nadam": th.optim.NAdam,
    "radam": th.optim.RAdam,
}


def build_datasets(datasets_config: dict[str, dict], device: str):
    datasets = {}
    for name, params in datasets_config.copy().items():
        dataset_type = params["type"].lower()
        dataset_params = {k: v for k, v in params.items() if k not in ["type", "split"]}
        datasets[name] = dataset_types[dataset_type](**dataset_params).to(device)
        if "split" in params.keys():
            splitted_dataset_names = params["split"].keys()
            splitted_dataset_ratios = params["split"].values()
            splitted_datasets = random_split(datasets[name], splitted_dataset_ratios)
            for splitted_dataset_name, splitted_dataset in zip(
                splitted_dataset_names, splitted_datasets
            ):
                datasets["_".join([name, splitted_dataset_name])] = splitted_dataset
    return datasets


def build_agents(agents_config: dict[str, dict], device: str):
    agents = {}
    for name, params in agents_config.items():
        agent_params = {k: v for k, v in params.items() if k != "type"}

        # build model
        if "model" in agent_params.keys():
            model_type = agent_params["model"]["type"].lower()
            model_params = {
                k: v for k, v in agent_params["model"].items() if k != "type"
            }
            agent_params["model"] = model_types[model_type](**model_params)

        # build optimizer
        if "optimizer" in agent_params.keys():
            optimizer_type = agent_params["optimizer"]["type"].lower()
            optimizer_params = {
                k: v for k, v in agent_params["optimizer"].items() if k != "type"
            }
            agent_params["optimizer"] = optimizer_types[optimizer_type]
            agent_params["optimizer_params"] = optimizer_params

        agents[name] = Agent(name=name, **agent_params).to(device)

    return agents


def build_tasks(
    tasks_config: dict[str, dict], agents: dict[str, Agent], datasets: dict, device: str
):
    tasks = {}
    for name, params in tasks_config.copy().items():
        task_type = params["type"].lower()
        task_params = {k: v for k, v in params.items() if k != "type"}

        # build network
        if "network" in task_params.keys():
            network_type = task_params["network"]["type"].lower()
            network_params = {
                k: v for k, v in task_params["network"].items() if k != "type"
            }
            task_params["network"] = network_types[network_type](
                agents=agents, **network_params
            )

        # build dataloader
        if "dataloader" in task_params.keys():
            dataloader_type = task_params["dataloader"]["type"].lower()
            dataloader_params = {
                k: v for k, v in task_params["dataloader"].items() if k != "type"
            }
            dataloader_params["dataset"] = datasets[dataloader_params["dataset"]]
            task_params["dataloader"] = dataloader_types[dataloader_type](
                **dataloader_params
            )

        # build losses
        for loss in [k for k in task_params.keys() if k.endswith("loss")]:
            loss_type = task_params[loss]["type"].lower()
            loss_params = {k: v for k, v in task_params[loss].items() if k != "type"}
            for baseline in [k for k in loss_params.keys() if k.endswith("baseline")]:
                baseline_type = loss_params[baseline]["type"].lower()
                baseline_params = {
                    k: v for k, v in loss_params[baseline].items() if k != "type"
                }
                loss_params[baseline] = baseline_types[baseline_type](**baseline_params)
            task_params[loss] = loss_types[loss_type](**loss_params).to(device)

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

    device = config["device"]

    datasets = build_datasets(config["datasets"], device)
    agents = build_agents(config["agents"], device)
    tasks = build_tasks(config["tasks"], agents, datasets, device)

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
