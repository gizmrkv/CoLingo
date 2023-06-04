from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def calculate(self, *args, **kwargs) -> dict:
        raise NotImplementedError
