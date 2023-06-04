from abc import ABC


class Callback(ABC):
    def on_begin(self):
        pass

    def on_pre_update(self):
        pass

    def on_update(self):
        pass

    def on_post_update(self):
        pass

    def on_end(self):
        pass
