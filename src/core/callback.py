from abc import ABC


class Callback(ABC):
    def on_begin(self):
        pass

    def on_update(self, step: int):
        pass

    def on_end(self):
        pass

    def on_early_stop(self, step: int):
        pass
