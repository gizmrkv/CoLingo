import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchtyping import TensorType


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        activations = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
        }
        self.activation = activations[activation]()

        if n_layers == 0:
            layers = []
        elif n_layers == 1:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                *[nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)],
                nn.Linear(hidden_dim, output_dim),
            ]

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: TensorType[..., "input_dim", float]
    ) -> TensorType[..., "output_dim", float]:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class IntSequenceMLPEncoder(nn.Module):
    def __init__(
        self,
        length: int,
        n_values: int,
        output_dim: int,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.length = length
        self.n_values = n_values
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.embed = nn.Embedding(n_values, embed_dim)
        self.mlp = MLP(length * embed_dim, output_dim, hidden_dim, n_layers, activation)

    def forward(
        self, input: TensorType[..., "length", int]
    ) -> TensorType[..., "output_dim", float]:
        embed = torch.cat([self.embed(input[:, i]) for i in range(self.length)], dim=1)
        output = self.mlp(embed)
        return output


class IntSequenceMLPDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        length: int,
        n_values: int,
        hidden_dim: int,
        n_layers: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.length = length
        self.n_values = n_values
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=length * n_values,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(
        self, latent: TensorType[..., "input_dim", float]
    ) -> TensorType[..., "length", "n_values", float]:
        logits = self.mlp(latent)
        logits = logits.view(-1, self.length, self.n_values)

        if self.training:
            distr = Categorical(logits=logits)
            return distr.sample(), logits
        else:
            return logits.argmax(dim=-1), logits
