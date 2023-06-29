import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torchtyping import TensorType


class ContMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation: str | nn.Module = "relu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        n_blocks: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.n_blocks = n_blocks

        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        self.activation = (
            activations[activation] if isinstance(activation, str) else activation
        )

        self.input2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation,
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_blocks)
            ]
        )

        if use_layer_norm:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(n_blocks)]
            )

    def forward(self, x: TensorType["batch", "input_dim", float]):
        x = self.input2hidden(x)
        for layer, norm in zip(self.mlp, self.layer_norms):
            if self.use_layer_norm:
                x = norm(x)

            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        x = self.hidden2output(x)
        return x
