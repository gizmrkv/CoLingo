import random

import torch as th


class Agent(th.nn.Module):
    """
    This class represents an Agent in the CoLingo library. Agents are
    responsible for processing the inputs from tasks and returning outputs. They
    have a specific model, an optimizer, a step probability, and optionally a name.

    Args:
        model (th.nn.Module): The model used by the agent to process inputs
            from tasks and return outputs.
        optimizer (th.optim.Optimizer): The optimizer used to adjust the
            parameters of the model in order to reduce the loss.
        optimizer_params (dict): The parameters used to configure the optimizer.
        step_prob (float, optional): The probability of the agent to take a step
            during the execution of a task. Defaults to 1.0, meaning the agent
            will always take a step.
        name (str | None, optional): The name of the agent. If not specified,
            the agent will be unnamed. Defaults to None.

    Methods:
        forward(*args, **kwargs): Executes a forward pass through the agent's
            model with the given arguments.
        step(): If the drawn random number is less than `step_prob`, performs
            an optimization step to adjust the model's parameters.
    """

    def __init__(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        optimizer_params: dict,
        step_prob: float = 1.0,
        name: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.step_prob = step_prob
        self.name = name

    def forward(self, *args, **kwargs):
        """
        Executes a forward pass through the agent's model with the given arguments.
        The specific behavior of this method depends on the implementation of the
        agent's model.

        Returns:
            The output of the model's forward pass. The type and content of this
            output depends on the model's implementation.
        """
        return self.model(*args, **kwargs)

    def step(self):
        """
        If the drawn random number is less than `step_prob`, performs an optimization
        step to adjust the model's parameters. This step is performed using the
        optimizer and its parameters that were specified when the agent was constructed.
        """
        if random.random() < self.step_prob:
            self.optimizer.step()
