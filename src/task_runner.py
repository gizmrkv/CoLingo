from .callback import Callback


class TaskRunner:
    def __init__(self, tasks: dict[str, Callback]):
        self.tasks = tasks

    def run(self, n_iterations: int):
        for task_name, task in self.tasks.items():
            task.on_begin()

        for _ in range(n_iterations):
            for task_name, task in self.tasks.items():
                task.on_update()

        for task_name, task in self.tasks.items():
            task.on_end()
