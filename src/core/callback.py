class Callback:
    def __init__(self, activated: bool = True):
        self.activated = activated

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

    def on_early_stop(self, iteration: int):
        pass
