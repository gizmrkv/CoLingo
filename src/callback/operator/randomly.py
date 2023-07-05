import random

from ..callback import Callback
from .callback_operator import CallbackOperator


class RandomlyCallbackOperator(CallbackOperator):
    def __init__(self, *callbacks: Callback, probability: float):
        super().__init__(*callbacks)
        self._probability = probability

    def on_update(self, iteration: int):
        if random.random() < self._probability:
            for callback in self._callbacks:
                callback.on_update(iteration)


def randomly(*callbacks: Callback, probability: float):
    return RandomlyCallbackOperator(*callbacks, probability=probability)
