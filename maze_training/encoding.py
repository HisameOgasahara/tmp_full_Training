"""Encode grids and queries with a small explicit vocabulary, without a tokenizer."""

import numpy as np
from .environment import WALL


class MazeEncoder:
    def __init__(self, max_size):
        self.max_size = max_size
        self.position_offset = 5
        self.position_count = max_size ** 2
        self.action_offset = self.position_offset + self.position_count
        self.size_offset = self.action_offset + 4
        self.world_token = self.size_offset + max_size + 1
        self.policy_token = self.world_token + 1
        self.query_token = self.world_token + 2
        self.vocabulary_size = self.query_token + 1
        self.context_length = self.position_count + 5

    def position_index(self, position):
        return position[0] * self.max_size + position[1]

    def decode_position(self, index):
        return divmod(int(index), self.max_size)

    def encode_grid(self, maze):
        grid = np.full((self.max_size, self.max_size), WALL, dtype=np.int64)
        grid[:maze.size, :maze.size] = maze.grid
        return grid.flatten().tolist()

    def encode_query(self, maze, position, action=None):
        task = self.policy_token if action is None else self.world_token
        tokens = [task, self.size_offset + maze.size, *self.encode_grid(maze), self.position_offset + self.position_index(position)]
        if action is not None:
            tokens.append(self.action_offset + action)
        return tokens + [self.query_token]

    @property
    def action_slice(self):
        return slice(self.action_offset, self.action_offset + 4)

    @property
    def position_slice(self):
        return slice(self.position_offset, self.position_offset + self.position_count)
