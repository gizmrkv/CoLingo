class Trainer:
    def __init__(self, games):
        self.games = games

    def train(self, n_epochs: int, print_interval: int = 100):
        for epoch in range(n_epochs):
            reward = self.games["lewis1"].play()

            if epoch % print_interval == 0:
                print(f"epoch: {epoch}, reward: {reward:.6f}")
