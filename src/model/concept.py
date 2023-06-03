import torch as th
from torchtyping import TensorType


class ConceptEncoder(th.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        embed_dim: int,
        concept_embed_dim: int,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.embed_dim = embed_dim
        self.concept_embed_dim = concept_embed_dim

        self.cpt_embeds = th.nn.ModuleList(
            [th.nn.Embedding(n_values, concept_embed_dim) for _ in range(n_attributes)]
        )
        self.embed = th.nn.Sequential(
            th.nn.Linear(n_attributes * concept_embed_dim, embed_dim),
        )

    def forward(self, x: TensorType["batch", "n_attributes", int]):
        x = th.cat(
            [self.cpt_embeds[i](x[:, i]) for i in range(self.n_attributes)],
            dim=1,
        )
        x = self.embed(x)
        return x


class ConceptDecoder(th.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.fc = th.nn.Sequential(
            th.nn.Linear(embed_dim, hidden_dim),
            th.nn.ReLU(),
            th.nn.LayerNorm(hidden_dim),
            th.nn.Linear(hidden_dim, n_attributes * n_values),
        )

    def forward(self, x: TensorType["batch", "embed_dim", float]):
        x = self.fc(x)
        x = x.view(-1, self.n_attributes, self.n_values)
        return x
