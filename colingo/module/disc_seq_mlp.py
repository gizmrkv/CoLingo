import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torchtyping import TensorType


class DiscSeqMLPEncoder(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    OUTPUT_DIM = "output_dim"

    def __init__(
        self,
        length: int,
        n_values: int,
        output_dim: int,
        hidden_dim: int,
        embed_dim: int,
        activation: str | nn.Module = "relu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        n_blocks: int = 1,
    ):
        super().__init__()
        self.length = length
        self.n_values = n_values
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
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
        self.embed = nn.Embedding(n_values, embed_dim)
        self.input2hidden = nn.Linear(length * embed_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

        self.tower = nn.ModuleList(
            [
                nn.Sequential(
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

    def forward(
        self, x: TensorType[BATCH, LENGTH, int]
    ) -> TensorType[BATCH, OUTPUT_DIM, float]:
        x = torch.cat([self.embed(x[:, i]) for i in range(self.length)], dim=1)
        x = self.input2hidden(x)
        for i, block in enumerate(self.tower):
            if self.use_residual:
                x = x + block(x)
            else:
                x = block(x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        x = self.activation(x)
        x = self.hidden2output(x)

        return x


class DiscSeqMLPDecoder(nn.Module):
    BATCH = "batch"
    INPUT_DIM = "input_dim"
    LENGTH = "length"
    N_VALUES = "n_values"

    def __init__(
        self,
        length: int,
        n_values: int,
        input_dim: int,
        hidden_dim: int,
        activation: str | nn.Module = "relu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        n_blocks: int = 1,
    ):
        super().__init__()
        self.length = length
        self.n_values = n_values
        self.input_dim = input_dim
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

        self.input2output = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, length * n_values),
        )

    def forward(
        self, x: TensorType[BATCH, INPUT_DIM, float]
    ) -> tuple[
        TensorType[BATCH, LENGTH, int], TensorType[BATCH, LENGTH, N_VALUES, float]
    ]:
        x = self.input2output(x)
        logits = x.view(-1, self.length, self.n_values)

        if self.training:
            distr = Categorical(logits=logits)
            x = distr.sample()
            return x, logits
        else:
            return logits.argmax(dim=-1), logits
