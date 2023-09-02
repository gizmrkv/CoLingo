import random
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from .core import Loggable, Stoppable, Task


class DictStopper(Task, Loggable[Mapping[str, Any]], Stoppable):
    def __init__(self, pred: Callable[[Mapping[str, Any]], bool]) -> None:
        self.pred = pred
        self.inputs: Dict[str, float] = {}
        self._stop = False

    def log(self, input: Mapping[str, Any]) -> None:
        self.inputs.update(input)

    def stop(self, step: int) -> bool:
        return self._stop

    def on_update(self, step: int) -> None:
        self._stop = self.pred(self.inputs)
        self.inputs.clear()

    def on_begin(self) -> None:
        self.inputs.clear()

    def on_end(self) -> None:
        self.inputs.clear()


class KeyChecker(Task, Loggable[Mapping[str, Any]]):
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def log(self, input: Mapping[str, Any]) -> None:
        for key in input:
            if key in self.seen:
                raise ValueError(f"Duplicate key: {key}")
            self.seen.add(key)

    def on_begin(self) -> None:
        self.seen.clear()

    def on_update(self, step: int) -> None:
        self.seen.clear()

    def on_end(self) -> None:
        self.seen.clear()


class StepCounter(Task):
    def __init__(self, loggers: Iterable[Loggable[Mapping[str, int]]]) -> None:
        self.loggers = loggers

    def on_update(self, step: int) -> None:
        for logger in self.loggers:
            logger.log({"step": step})


class Stopwatch(Task):
    def __init__(self, loggers: Iterable[Loggable[Mapping[str, float]]]) -> None:
        super().__init__()
        self.loggers = loggers
        self.start_time = time.time()

    def on_update(self, step: int) -> None:
        elapsed_time = time.time() - self.start_time
        for logger in self.loggers:
            logger.log({"elapsed_time": elapsed_time})


class TimeDebugger(Task):
    def __init__(self, tasks: Iterable[Task]) -> None:
        self.tasks = tasks

    def on_update(self, step: int) -> None:
        for i, task in enumerate(self.tasks):
            torch.cuda.synchronize()
            start = time.time()
            task.on_update(step)
            torch.cuda.synchronize()
            end = time.time()
            print(f"{i}st time: {end - start:.3f} sec")
