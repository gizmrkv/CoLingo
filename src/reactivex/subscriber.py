from abc import ABC, abstractmethod


class Subscriber(ABC):
    @abstractmethod
    def on_next(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_completed(self):
        pass

    @abstractmethod
    def on_error(self, error):
        pass
