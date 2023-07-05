from ..callback import Callback
from .callback_operator import CallbackOperator


class NeverCallbackOperator(CallbackOperator):
    def __init__(self, *callbacks: Callback):
        super().__init__(*callbacks)

    def on_update(self, iteration: int):
        pass


def never(*callbacks: Callback):
    return NeverCallbackOperator(*callbacks)
