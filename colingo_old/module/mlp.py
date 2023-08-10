import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torchtyping import TensorType


class MLPEncoder(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    OUTPUT_DIM = "output_dim"

    def __init__(
        self,
        length: int,
        n_values: int,
        output_dim: int,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self._length = length
        self._n_values = n_values
        self._output_dim = output_dim
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._activation = activation

        self._embed = nn.Embedding(n_values, embed_dim)

        self._mlp = MLP(
            input_dim=length * embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(
        self, x: TensorType[BATCH, LENGTH, int]
    ) -> TensorType[BATCH, OUTPUT_DIM, float]:
        x = torch.cat([self._embed(x[:, i]) for i in range(self._length)], dim=1)
        return self._mlp(x)


class MLPDecoder(nn.Module):
    BATCH = "batch"
    INPUT_DIM = "input_dim"
    LENGTH = "length"
    N_VALUES = "n_values"

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
        self._input_dim = input_dim
        self._length = length
        self._n_values = n_values
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._activation = activation

        self._mlp = MLP(
            input_dim=input_dim,
            output_dim=length * n_values,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(
        self, x: TensorType[BATCH, INPUT_DIM, float]
    ) -> tuple[
        TensorType[BATCH, LENGTH, int], TensorType[BATCH, LENGTH, N_VALUES, float]
    ]:
        x = self._mlp(x)
        logits = x.view(-1, self._length, self._n_values)

        if self.training:
            distr = Categorical(logits=logits)
            x = distr.sample()
            return x, logits
        else:
            return logits.argmax(dim=-1), logits


class MLP(nn.Module):
    BATCH = "batch"
    INPUT_DIM = "input_dim"
    OUTPUT_DIM = "output_dim"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
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

        self._layers = nn.ModuleList(layers)

    def forward(
        self, x: TensorType[BATCH, INPUT_DIM, float]
    ) -> TensorType[BATCH, OUTPUT_DIM, float]:
        for layer in self._layers[:-1]:
            x = self.activation(layer(x))
        x = self._layers[-1](x)
        return x
