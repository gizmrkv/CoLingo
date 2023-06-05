from abc import ABC


class Callback(ABC):
    def on_begin(self):
        pass

    def on_pre_update(self, iteration: int):
        pass

    def on_update(self, iteration: int):
        pass

    def on_post_update(self, iteration: int):
        pass

    def on_end(self):
        pass
