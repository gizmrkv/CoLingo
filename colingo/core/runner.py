from typing import Iterable

import tqdm


class RunnerCallback:
    """
    Base class for defining callbacks during the execution of a Runner.
    """

    def on_begin(self) -> None:
        """
        Called at the beginning of the Runner's execution.
        """
        pass

    def on_update(self, step: int) -> None:
        """
        Called at each step during the execution of the Runner.

        Args:
            step (int): The current step number.
        """
        pass

    def on_end(self) -> None:
        """
        Called at the end of the Runner's execution.
        """
        pass


class EarlyStopper:
    """
    Class to handle early stopping based on a stopping condition.
    """

    def stop(self, step: int) -> bool:
        """
        Determine if early stopping should be performed at the current step.

        Args:
            step (int): The current step number.

        Returns:
            bool: True if early stopping should be performed, False otherwise.
        """
        return False


class Runner:
    """
    Class to manage the execution of tasks with callbacks and early stopping.

    Args:
        callbacks (Iterable[RunnerCallback]): RunnerCallback instances.
        stopper (EarlyStopper, optional): EarlyStopper instance for controlling early stopping. Defaults to None.
        use_tqdm (bool, optional): Whether to display progress using tqdm. Defaults to True.
    """

    def __init__(
        self,
        callbacks: Iterable[RunnerCallback],
        stopper: EarlyStopper | None = None,
        use_tqdm: bool = True,
    ):
        self.callbacks = callbacks
        self.use_tqdm = use_tqdm
        self.stopper = stopper or EarlyStopper()

    def run(self, n_steps: int) -> None:
        """
        Execute the tasks with callbacks and early stopping.

        Args:
            n_steps (int): Number of steps to run the tasks.
        """
        for callback in self.callbacks:
            callback.on_begin()

        rg: Iterable[int] = range(n_steps)
        if self.use_tqdm:
            rg = tqdm.tqdm(rg)

        for step in rg:
            if self.stopper.stop(step):
                break

            for callback in self.callbacks:
                callback.on_update(step)

        for callback in self.callbacks:
            callback.on_end()
