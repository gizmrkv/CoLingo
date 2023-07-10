from .logger import Logger


class EarlyStopper(Logger):
    def __init__(self, metric: str, threshold: float):
        self._metric = metric
        self._threshold = threshold
        self._stop = False

    def log(self, log: dict):
        if log[self._metric] >= self._threshold:
            self._stop = True

    def __call__(self):
        return self._stop
