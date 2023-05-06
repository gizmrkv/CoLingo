import datetime
import random
from pprint import pprint

import toml
import torch as th
import yaml
from torch.utils.data import DataLoader

from src.agent import Agent
from src.baseline import BatchMeanBaseline, MeanBaseline
from src.callback import Callback
from src.dataset import (
    build_concept_dataset,
    build_normal_dataset,
    build_onehot_concept_dataset,
    random_split,
)
from src.evaluator import ModuleEvaluator, SignalingEvaluator
from src.logger import ConsoleLogger, WandBLogger
from src.loss import ConceptLoss, OnehotConceptLoss, ReinforceLoss
from src.model import (
    EmbeddingConceptSequentialMessageModel,
    OnehotConceptSequntialMessageModel,
    OnehotConceptSymbolMessageModel,
)
from src.network import CustomNetwork, Network
from src.task import AgentSaver, SignalingTrainer
from src.task_runner import TaskRunner
from src.util import find_length, fix_seed


class ConceptAccuracy:
    def __init__(self, n_attributes: int, n_values: int):
        self.n_attributes = n_attributes
        self.n_values = n_values

    def __call__(self, dataset: th.Tensor, output: th.Tensor):
        batch_size = dataset.shape[0]
        output = (
            output.view(batch_size * self.n_attributes, -1)
            .argmax(dim=-1)
            .reshape(-1, self.n_attributes)
        )
        acc = (output == dataset).float().mean().item()
        return acc


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

    n_attributes = config["onehots_config"]["n_attributes"]
    n_values = config["onehots_config"]["n_values"]
    evaluators = {
        "acc": lambda dataset, message, output, aux_s, aux_r: ConceptAccuracy(
            n_attributes, n_values
        )(dataset, output),
    }
    signaling_evaluator = SignalingEvaluator(
        agents["agent1"],
        agents["agent2"],
        datasets["dataset1"],
        evaluators,
        [WandBLogger(project="hoge", name="fuga"), ConsoleLogger()],
        interval=1,
    )
    tasks["signaling_evaluator"] = signaling_evaluator

    tasks["model_saver"] = AgentSaver(agents, 1000, f"{exp_dir}/models")

    runner = TaskRunner(tasks)
    runner.run(config["n_iterations"])


if __name__ == "__main__":
    with open("config/sample.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(config)
