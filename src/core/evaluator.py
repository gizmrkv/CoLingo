import torch as th
from networkx import DiGraph

from .agent import Agent
from .callback import Callback
from .logger import Logger


class LanguageEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: DiGraph,
        dataset: th.Tensor,
        metrics: dict[str, callable],
        loggers: dict[str, Logger],
        interval: int = 10,
        name: str = "eval",
    ):
        super().__init__()
        self.agents = agents
        self.network = network
        self.dataset = dataset
        self.metrics = metrics
        self.loggers = loggers
        self.interval = interval
        self.name = name

        self._nodes = list(self.network.nodes)
        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        languages = {}
        for agent_name in self._nodes:
            agent = self.agents[agent_name]
            agent.eval()
            with th.no_grad():
                lang, _ = agent(self.dataset, "sender")

            languages[agent_name] = lang

        logs = {}
        for metric_name, metric in self.metrics.items():
            value = metric(concept=self.dataset, languages=languages)
            logs[metric_name] = value

        for logger in self.loggers.values():
            logger.log({self.name: logs})
