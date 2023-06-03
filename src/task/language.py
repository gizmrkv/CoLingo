from typing import Callable, Iterable

import torch as th
from networkx import DiGraph
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.logger import Logger
from ..core.network import generate_custom_graph


class LanguageEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        dataloader: DataLoader,
        metrics: dict[str, Callable],
        loggers: Iterable[Logger],
        name: str,
        network: DiGraph | None = None,
    ):
        super().__init__()
        self.agents = agents
        self.dataloader = dataloader
        self.metrics = metrics
        self.loggers = loggers
        self.name = name
        self.network = network

        if self.network is None:
            self.network = generate_custom_graph(list(self.agents.keys()))
        self._nodes = list(self.network.nodes)

    def on_end(self):
        for input, target in self.dataloader:
            languages = {}
            lengths = {}
            for agent_name in self._nodes:
                agent = self.agents[agent_name]
                agent.eval()
                with th.no_grad():
                    lang, _, _, _ = agent(input, self.command)

                languages[agent_name] = lang
                lengths[agent_name] = th.argmin(lang, dim=1) + 1

            logs = {}
            for metric_name, metric in self.metrics.items():
                value = metric(
                    input=input, languages=languages, lengths=lengths, target=target
                )
                logs[metric_name] = value

            for logger in self.loggers:
                logger.log({self.name: logs})
