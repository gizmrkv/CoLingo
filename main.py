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
from src.core.dataset import (build_concept_dataset, build_normal_dataset,
                              build_onehot_concept_dataset, random_split)
from src.core.evaluator import LanguageEvaluator
from src.core.logger import ConsoleLogger, WandBLogger
from src.core.loss import ConceptLoss, OnehotConceptLoss, ReinforceLoss
from src.core.metric import (ConceptAccuracy, LanguageSimilarity,
                             MessageEntropy, MessageLength,
                             TopographicSimilarity)
from src.core.network import create_custom_graph
from src.core.task_runner import TaskRunner
from src.core.util import AgentSaver, fix_seed
from src.model.internal_representation import InternalRepresentaionModel
from src.model.misc import (EmbeddingConceptSequentialMessageModel,
                            OnehotConceptSequntialMessageModel,
                            OnehotConceptSymbolMessageModel)
from src.task.identity import IdentityEvaluator, IdentityTrainer
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
    "identity": IdentityTrainer,
    "signaling_eval": SignalingEvaluator,
    "identity_eval": IdentityEvaluator,
    "language_eval": LanguageEvaluator,
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
}
logger_types = {"console": ConsoleLogger, "wandb": WandBLogger}


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
            task_params["network"] = network_types[network_type](**network_params)

        # assign dataset
        if "dataset" in task_params.keys():
            task_params["dataset"] = datasets[task_params["dataset"]]

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

        # build metrics
        if "metrics" in task_params.keys():
            for metric_name, metric in task_params["metrics"].items():
                metric_type = metric["type"].lower()
                metric_params = {k: v for k, v in metric.items() if k != "type"}
                task_params["metrics"][metric_name] = metric_types[metric_type](
                    **metric_params
                )

        # build logger
        if "loggers" in task_params.keys():
            for logger_name, logger in task_params["loggers"].items():
                logger_type = logger["type"].lower()
                logger_params = {k: v for k, v in logger.items() if k != "type"}
                task_params["loggers"][logger_name] = logger_types[logger_type](
                    **logger_params
                )

        # build task
        tasks[name] = task_types[task_type](agents=agents, name=name, **task_params)

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

    tasks["model_saver"] = AgentSaver(agents, 1000, f"{exp_dir}/models")

    runner = TaskRunner(tasks)
    runner.run(config["n_iterations"])


if __name__ == "__main__":
    # config_path = sys.argv[1]
    config_path = "config/sample.yaml"

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
