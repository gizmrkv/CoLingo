import datetime
import json
import pickle
import random
import sys

import toml
import torch as th
import yaml

from src.core.agent import Agent
from src.core.baseline import BatchMeanBaseline, MeanBaseline
from src.core.command import Command
from src.core.dataset import (build_concept_dataset, build_normal_dataset,
                              build_onehot_concept_dataset, random_split)
from src.core.evaluator import LanguageEvaluator
from src.core.logger import ConsoleLogger, WandBLogger
from src.core.loss import ConceptLoss, OnehotConceptLoss, ReinforceLoss
from src.core.metric import (ConceptAccuracy, LanguageSimilarity,
                             MessageEntropy, MessageLength, SignalingDisplay,
                             TopographicSimilarity, UniqueMessage)
from src.core.network import create_custom_graph
from src.core.task_runner import TaskRunner
from src.core.task_scheduler import LinearTaskScheduler
from src.core.util import AgentSaver, fix_seed
from src.model.internal_representation import InternalRepresentaionModel
from src.model.misc import (EmbeddingConceptSequentialMessageModel,
                            OnehotConceptSequntialMessageModel,
                            OnehotConceptSymbolMessageModel)
from src.task.prediction import PredictionEvaluator, PredictionTrainer
from src.task.signaling import SignalingEvaluator, SignalingTrainer

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
    "internal": InternalRepresentaionModel,
}
loss_types = {
    "reinforce": ReinforceLoss,
    "concept": ConceptLoss,
    "onehot_concept": OnehotConceptLoss,
}
baseline_types = {"mean": MeanBaseline, "batch_mean": BatchMeanBaseline}
task_types = {
    "signaling": SignalingTrainer,
    "prediction": PredictionTrainer,
    "signaling_eval": SignalingEvaluator,
    "prediction_eval": PredictionEvaluator,
    "language_eval": LanguageEvaluator,
    "linear_scheduler": LinearTaskScheduler,
}
network_types = {"custom": create_custom_graph}
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
metric_types = {
    "concept_accuracy": ConceptAccuracy,
    "topsim": TopographicSimilarity,
    "langsim": LanguageSimilarity,
    "msglen": MessageLength,
    "msgent": MessageEntropy,
    "unique": UniqueMessage,
    "signdisp": SignalingDisplay,
}
logger_types = {"console": ConsoleLogger, "wandb": WandBLogger}


def create_instance(types: dict, type: str, **params):
    if type not in types.keys():
        raise ValueError(f"Invalid type: {type}")

    return types[type](**params)


def create_datasets(datasets_config: dict[str, dict], device: str):
    datasets = {}
    for name, params in datasets_config.items():
        split = params.get("split", None)
        params.pop("split", None)
        datasets[name] = create_instance(dataset_types, device=device, **params)

        if split is not None:
            splitted_names = split.keys()
            splitted_ratios = split.values()
            splitted_datasets = random_split(datasets[name], splitted_ratios)
            for splitted_name, splitted in zip(splitted_names, splitted_datasets):
                datasets["_".join([name, splitted_name])] = splitted

    return datasets


def create_agent(name: str | None = None, device: str | None = None, **params):
    params["model"] = create_instance(model_types, **params["model"]).to(device)
    params["optimizer_params"] = params["optimizer"].copy()
    params["optimizer_params"].pop("type")
    params["optimizer"] = optimizer_types[params["optimizer"]["type"]]
    params["name"] = name
    return Agent(**params)


def create_agents(agents: dict[str, dict], device: str):
    return {name: create_agent(name, device, **agent) for name, agent in agents.items()}


def create_task(
    types: dict,
    type: str,
    name: str | None = None,
    datasets: dict | None = None,
    agents: dict | None = None,
    device: str | None = None,
    **params,
):
    if type.endswith("scheduler"):
        params["task"] = create_task(
            types,
            name=name,
            datasets=datasets,
            agents=agents,
            device=device,
            **params["task"],
        )
        scheduler = create_instance(types, type, **params)
        return scheduler

    if "network" in params.keys():
        params["network"] = create_instance(network_types, **params["network"])

    if "dataloader" in params.keys():
        params["dataloader"]["dataset"] = datasets[params["dataloader"]["dataset"]]
        params["dataloader"] = create_instance(dataloader_types, **params["dataloader"])

    for loss in [k for k in params.keys() if k.endswith("loss")]:
        for baseline in [k for k in params[loss].keys() if k.endswith("baseline")]:
            params[loss][baseline] = create_instance(
                baseline_types, **params[loss][baseline]
            )
        params[loss] = create_instance(loss_types, **params[loss]).to(device)

    if "metrics" in params.keys():
        for metric_name, metric in params["metrics"].items():
            params["metrics"][metric_name] = create_instance(metric_types, **metric)

    if "loggers" in params.keys():
        for logger_name, logger in params["loggers"].items():
            params["loggers"][logger_name] = create_instance(logger_types, **logger)

    for command in [k for k in params.keys() if k.endswith("command")]:
        if isinstance(params[command], str):
            params[command] = Command[params[command].upper()]
        elif isinstance(params[command], int):
            params[command] = Command(params[command])
        else:
            raise ValueError("command must be either str or int")

    params["agents"] = agents
    params["name"] = name

    return create_instance(types, type, **params)


def create_tasks(
    tasks: dict[str, dict],
    datasets: dict | None = None,
    agents: dict | None = None,
    device: str | None = None,
):
    return {
        name: create_task(
            task_types,
            name=name,
            datasets=datasets,
            agents=agents,
            device=device,
            **task,
        )
        for name, task in tasks.items()
    }


def check_config(datasets, agents, tasks):
    # TODO: implemnet
    pass


def main(config: dict):
    date = datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
    exp_id = random.randint(0, 100000)
    exp_dir = f"exp/{config['exp_name']} {date} {exp_id}"

    fix_seed(config["seed"])

    device = config["device"]

    datasets = create_datasets(config["datasets"], device)
    agents = create_agents(config["agents"], device)
    tasks = create_tasks(
        config["tasks"], datasets=datasets, agents=agents, device=device
    )

    check_config(datasets, agents, tasks)

    tasks["model_saver"] = AgentSaver(agents, 1000, f"{exp_dir}/models")

    runner = TaskRunner(tasks.values())
    runner.run(config["n_iterations"])


if __name__ == "__main__":
    # config_path = sys.argv[1]
    config_path = "config/sample.yaml"
    # config_path = "config/small.yaml"

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif config_path.endswith(".toml"):
            config = toml.load(f)
        elif config_path.endswith(".pickle"):
            config = pickle.load(f)
        else:
            raise ValueError(f"Unknown file extension: {config_path}")

    main(config)
