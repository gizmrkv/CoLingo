from .game import Game
import math


class Trainer:
    def __init__(self, games: dict[str, Game]):
        self.games = games

    def train(self, n_epochs: int):
        for epoch in range(n_epochs):
            for game_name, game in self.games.items():
                game.play_pool, play_count = math.modf(game.play_pool + game.play_rate)
                for _ in range(int(play_count)):
                    game.play()
