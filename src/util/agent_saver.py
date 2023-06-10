import os

import torch as th

from ..agent import Agent
from ..core import Callback


class AgentSaver(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        path: str,
    ):
        super().__init__()
        self.agents = agents
        self.path = path

    def on_update(self, iteration: int):
        for agent_name, agent in self.agents.items():
            save_dir = f"{self.path}/{agent_name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            th.save(
                agent,
                f"{save_dir}/{iteration}.pth",
            )
