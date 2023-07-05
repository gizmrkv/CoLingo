from ..callback import Callback


class CallbackOperator(Callback):
    def __init__(self, *callbacks: Callback):
        self._callbacks = callbacks

    def on_begin(self):
        for callback in self._callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self._callbacks:
            callback.on_end()

    def on_early_stop(self, iteration: int):
        for callback in self._callbacks:
            callback.on_early_stop(iteration)
