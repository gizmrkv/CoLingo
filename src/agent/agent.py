from abc import ABC, abstractmethod

import torch as th


class Agent(ABC, th.nn.Module):
    @abstractmethod
    def input(self, inputs: dict):
        raise NotImplementedError

    @abstractmethod
    def output(*outputs, hidden):
        raise NotImplementedError
