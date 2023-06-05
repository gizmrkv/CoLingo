from typing import Iterable

import torch as th

from ..agent import Agent
from ..core.callback import Callback
from ..logger import Logger
from ..metric import Metric


class LanguageEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        input: th.Tensor,
        metrics: Iterable[Metric],
        loggers: Iterable[Logger],
        input_key,
        output_key,
        name: str,
    ):
        super().__init__()
        self.agents = agents
        self.input = input
        self.metrics = metrics
        self.loggers = loggers
        self.input_key = input_key
        self.output_key = output_key
        self.name = name

        self.agent_names = list(self.agents.keys())

    def on_end(self):
        languages = {}
        lengths = {}
        for agent_name in self.agent_names:
            agent = self.agents[agent_name]
            agent.eval()
            with th.no_grad():
                hidden = agent.input({self.input_key: self.input})
                ((lang, _, _, _),) = agent.output(self.output_key, hidden=hidden)

            languages[agent_name] = lang.cpu().numpy()
            lengths[agent_name] = lang.cpu().numpy().argmin(axis=-1) + 1

        logs = {}
        for metric in self.metrics:
            met = metric.calculate(
                input=self.input, languages=languages, lengths=lengths
            )
            logs |= met

        for logger in self.loggers:
            logger.log({self.name: logs}, flush=True)
