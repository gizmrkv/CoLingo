import torch as th


class SingleWordModel(th.nn.Module):
    def __init__(
        self, n_attributes: int, n_values: int, vocab_size: int, hidden_size: int = 64
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.fc1 = th.nn.Sequential(
            th.nn.Linear(n_attributes * n_values, hidden_size),
            th.nn.ReLU(),
            th.nn.Linear(hidden_size, vocab_size),
        )
        self.fc2 = th.nn.Sequential(
            th.nn.Linear(vocab_size, hidden_size),
            th.nn.ReLU(),
            th.nn.Linear(hidden_size, n_attributes * n_values),
        )

    def forward(self, x: th.Tensor, input_type: str, is_training: bool = False):
        log_prob = None
        entropy = None
        if input_type == "object":
            x = self.fc1(x)
            if is_training:
                dist = th.distributions.Categorical(logits=x)
                x = dist.sample()
                log_prob = dist.log_prob(x).mean()
                entropy = dist.entropy().mean()
            else:
                x = x.argmax(dim=-1)

            x = th.nn.functional.one_hot(x, self.vocab_size).float()
        elif input_type == "message":
            x = self.fc2(x)
            batch_size = x.shape[0]
            x = x.view(batch_size * self.n_attributes, self.n_values)

        return x, log_prob, entropy


def build_model(model_type: str, model_args: dict) -> th.nn.Module:
    models_dict = {"single_word": SingleWordModel}
    return models_dict[model_type](**model_args)
