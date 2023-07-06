from abc import ABC, abstractmethod

from .subscriber import Subscriber


class Publisher(ABC):
    @abstractmethod
    def subscribe(self, subscriber: Subscriber):
        pass
