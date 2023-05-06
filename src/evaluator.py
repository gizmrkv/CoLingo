import torch as th

from .callback import Callback
from .logger import Logger


class ModuleEvaluator(Callback):
    def __init__(
        self,
        module: th.nn.Module,
        dataset: th.Tensor,
        evaluator: dict,
        logger: Logger,
        interval: int = 10,
        name: str = "eval",
    ):
        super().__init__()
        self.module = module
        self.dataset = dataset
        self.evaluator = evaluator
        self.logger = logger
        self.interval = interval
        self.name = name

        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        self.module.eval()
        output = self.module(self.dataset)
        logs = {}
        for name, func in self.evaluator.items():
            value = func(self.dataset, output)
            logs[name] = value
        self.logger.log({self.name: logs})


class SignalingEvaluator(Callback):
    def __init__(
        self,
        sender: th.nn.Module,
        receiver: th.nn.Module,
        dataset: th.Tensor,
        evaluator: dict,
        loggers: list[Logger],
        interval: int = 10,
        name: str = "eval",
    ):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.dataset = dataset
        self.evaluator = evaluator
        self.loggers = loggers
        self.interval = interval
        self.name = name

        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        for agent in [self.sender, self.receiver]:
            agent.eval()
        message, aux_s = self.sender(self.dataset, "sender")
        output, aux_r = self.receiver(message, "receiver")
        logs = {}
        for name, func in self.evaluator.items():
            value = func(self.dataset, message, output, aux_s, aux_r)
            logs[name] = value

        for logger in self.loggers:
            logger.log({self.name: logs})
